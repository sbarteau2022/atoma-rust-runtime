//! Deterministic arithmetic operator (ported 2026-10-19).
//!
//! PROVENANCE: the evaluator is a Rust port of the Elle worker's
//! `src/calc.ts` — a hand-rolled recursive-descent arithmetic evaluator
//! ("LLMs are unreliable at exact arithmetic on real numbers. A
//! deterministic calculator closes that gap."). Same grammar, same
//! fail-closed discipline: anything that does not parse as a pure
//! arithmetic expression is declined, never guessed.
//!
//! Monster-server addition: a natural-language front end resolves number
//! words (`"two"`, `"twenty one"`, `"one hundred and five"`) and operator
//! words (`"plus"`, `"minus"`, `"times"`, `"divided by"`, `"squared"`) to a
//! symbolic expression before parsing. The result is an exact computation,
//! not a retrieval — callers should serve it without trajectory gating.

use super::benchmark_operators::OperatorExecutionSynthesis;

/// Entry point: return a deterministic synthesis for bare arithmetic
/// prompts, or `None` to decline (prompt is not pure arithmetic).
pub fn try_evaluate_deterministic_arithmetic(prompt: &str) -> Option<OperatorExecutionSynthesis> {
    let expr = normalize_to_expression(prompt)?;
    let value = eval_expression(&expr).ok()?;
    if !value.is_finite() {
        return None;
    }
    let answer = if (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        let s = format!("{:.6}", value);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    Some(OperatorExecutionSynthesis {
        operator_family: "DeterministicArithmeticOperator".to_string(),
        canonical_skeleton: Some(expr.clone()),
        synthesized_response: format!("{} = {}.", pretty(&expr), answer),
        mlir_ssa_ops_executed: 0,
        mlir_result_q32: None,
        episodes_indexed: 0,
    })
}

/// "2+2" -> "2 + 2" for display.
fn pretty(expr: &str) -> String {
    let mut out = String::with_capacity(expr.len() + 8);
    for ch in expr.chars() {
        if "+-*/%^()".contains(ch) {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            out.push(ch);
            out.push(' ');
        } else {
            out.push(ch);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── natural-language front end ────────────────────────────────────────────

pub(crate) fn number_word_value(w: &str) -> Option<(f64, WordKind)> {
    let v = match w {
        "zero" => (0.0, WordKind::Small),
        "one" => (1.0, WordKind::Small),
        "two" => (2.0, WordKind::Small),
        "three" => (3.0, WordKind::Small),
        "four" => (4.0, WordKind::Small),
        "five" => (5.0, WordKind::Small),
        "six" => (6.0, WordKind::Small),
        "seven" => (7.0, WordKind::Small),
        "eight" => (8.0, WordKind::Small),
        "nine" => (9.0, WordKind::Small),
        "ten" => (10.0, WordKind::Small),
        "eleven" => (11.0, WordKind::Small),
        "twelve" => (12.0, WordKind::Small),
        "thirteen" => (13.0, WordKind::Small),
        "fourteen" => (14.0, WordKind::Small),
        "fifteen" => (15.0, WordKind::Small),
        "sixteen" => (16.0, WordKind::Small),
        "seventeen" => (17.0, WordKind::Small),
        "eighteen" => (18.0, WordKind::Small),
        "nineteen" => (19.0, WordKind::Small),
        "twenty" => (20.0, WordKind::Small),
        "thirty" => (30.0, WordKind::Tens),
        "forty" => (40.0, WordKind::Tens),
        "fifty" => (50.0, WordKind::Tens),
        "sixty" => (60.0, WordKind::Tens),
        "seventy" => (70.0, WordKind::Tens),
        "eighty" => (80.0, WordKind::Tens),
        "ninety" => (90.0, WordKind::Tens),
        "hundred" => (100.0, WordKind::Hundred),
        "thousand" => (1_000.0, WordKind::Scale),
        "million" => (1_000_000.0, WordKind::Scale),
        // Implicit-constant lexicon (2026-10-20, Stewart): pragmatic
        // arithmetic words are constants of the language.
        "pair" => (2.0, WordKind::Small),
        "weekly" => (7.0, WordKind::Small),
        "dozen" => (12.0, WordKind::Dozen),
        _ => return None,
    };
    Some(v)
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum WordKind {
    Small,
    Tens,
    Hundred,
    Scale,
    /// Multiplier constant: "two dozen" = 24, bare "dozen" = 12.
    Dozen,
}

/// Consume a maximal run of number words starting at `i`; return (value, next_i).
fn parse_number_phrase(atoms: &[String], i: usize) -> Option<(f64, usize)> {
    let mut total = 0.0;
    let mut current = 0.0;
    let mut j = i;
    let mut any = false;
    while j < atoms.len() {
        let w = atoms[j].as_str();
        if w == "and" {
            j += 1;
            continue;
        }
        let Some((v, kind)) = number_word_value(w) else {
            break;
        };
        any = true;
        match kind {
            WordKind::Small | WordKind::Tens => current += v,
            WordKind::Dozen => {
                if current == 0.0 {
                    current = 12.0;
                } else {
                    current *= 12.0;
                }
            }
            WordKind::Hundred => {
                if current == 0.0 {
                    current = 1.0;
                }
                current *= 100.0;
            }
            WordKind::Scale => {
                if current == 0.0 {
                    current = 1.0;
                }
                total += current * v;
                current = 0.0;
            }
        }
        j += 1;
    }
    if !any {
        return None;
    }
    Some((total + current, j))
}

/// Multi-word operator phrases, longest first. ("subtracted from" flips operands.)
fn phrase_operator(atoms: &[String], i: usize) -> Option<(&'static str, usize, bool)> {
    let rest: String = atoms[i..].join(" ");
    for (phrase, sym, flip) in [
        ("to the power of", "^", false),
        ("raised to", "^", false),
        ("multiplied by", "*", false),
        ("divided by", "/", false),
        ("subtracted from", "-", true),
        ("added to", "+", false),
    ] {
        if rest.starts_with(phrase)
            && rest[phrase.len()..]
                .chars()
                .next()
                .is_none_or(|c| c == ' ')
        {
            let nwords = phrase.split_whitespace().count();
            return Some((sym, nwords, flip));
        }
    }
    None
}

fn word_operator(w: &str) -> Option<&'static str> {
    match w {
        "plus" | "add" => Some("+"),
        "minus" | "subtract" => Some("-"),
        "times" | "multiply" | "multiplied" => Some("*"),
        "divided" | "divide" | "over" => Some("/"),
        "modulo" | "mod" => Some("%"),
        "power" | "pow" => Some("^"),
        _ => None,
    }
}

/// Split one whitespace-separated word into atoms: letter runs, digit runs,
/// and single punctuation marks. "sqrt(16)" -> ["sqrt","(","16",")"].
fn atomize(word: &str) -> Vec<String> {
    let mut atoms = Vec::new();
    let mut cur = String::new();
    let mut cur_kind: Option<char> = None; // 'a' | 'd'
    let flush = |atoms: &mut Vec<String>, cur: &mut String, cur_kind: &mut Option<char>| {
        if !cur.is_empty() {
            atoms.push(std::mem::take(cur));
        }
        *cur_kind = None;
    };
    for ch in word.chars() {
        if ch.is_ascii_alphabetic() {
            if cur_kind != Some('a') {
                flush(&mut atoms, &mut cur, &mut cur_kind);
                cur_kind = Some('a');
            }
            cur.push(ch);
        } else if ch.is_ascii_digit() || ch == '.' {
            if cur_kind != Some('d') {
                flush(&mut atoms, &mut cur, &mut cur_kind);
                cur_kind = Some('d');
            }
            cur.push(ch);
        } else {
            flush(&mut atoms, &mut cur, &mut cur_kind);
            atoms.push(ch.to_string());
        }
    }
    flush(&mut atoms, &mut cur, &mut cur_kind);
    atoms
}

fn is_digit_atom(a: &str) -> bool {
    !a.is_empty()
        && a.chars().all(|c| c.is_ascii_digit() || c == '.')
        && a.chars().any(|c| c.is_ascii_digit())
}

/// Strip question framing and reduce the prompt to a symbolic expression.
/// Returns None when the prompt is not pure arithmetic.
fn normalize_to_expression(prompt: &str) -> Option<String> {
    let mut text = prompt.trim().to_lowercase();
    // Strip leading question frames.
    for frame in ["what's ", "whats ", "what is ", "calculate ", "compute ", "evaluate ", "solve "] {
        if let Some(rest) = text.strip_prefix(frame) {
            text = rest.trim_start().to_string();
            break;
        }
    }
    text = text.trim_end_matches(['?', '.', '!']).trim().to_string();
    if text.is_empty() {
        return None;
    }
    let atoms: Vec<String> = text.split_whitespace().flat_map(atomize).collect();
    let mut tokens: Vec<String> = Vec::new();
    let mut i = 0;
    while i < atoms.len() {
        // Multi-word operator phrases.
        if let Some((sym, nwords, flip)) = phrase_operator(&atoms, i) {
            if flip {
                // "X subtracted from Y" -> "Y - X". X is the last emitted
                // token; Y is the number phrase or digit following the phrase.
                let left = tokens.pop()?;
                let right_i = i + nwords;
                if let Some((yval, next)) = parse_number_phrase(&atoms, right_i) {
                    tokens.push(format_num_tok(yval));
                    tokens.push(sym.to_string());
                    tokens.push(left);
                    i = next;
                    continue;
                } else if right_i < atoms.len() && is_digit_atom(&atoms[right_i]) {
                    tokens.push(atoms[right_i].clone());
                    tokens.push(sym.to_string());
                    tokens.push(left);
                    i = right_i + 1;
                    continue;
                }
                return None;
            }
            tokens.push(sym.to_string());
            i += nwords;
            continue;
        }
        let w = atoms[i].as_str();
        // Implicit-constant operators (2026-10-20): "twice" = x2,
        // "half of N" = N / 2.
        if w == "twice" {
            // Postfix: "<num> twice" -> "<num> * 2".
            let prev_is_operand = tokens.last().is_some_and(|t| {
                is_digit_atom(t) || t == ")" || t.parse::<f64>().is_ok()
            });
            if prev_is_operand {
                tokens.push("*".to_string());
                tokens.push("2".to_string());
                i += 1;
                continue;
            }
            // Prefix: "twice <num>" -> "( 2 * <num> )".
            if let Some((val, next)) = parse_number_phrase(&atoms, i + 1) {
                tokens.push("(".to_string());
                tokens.push("2".to_string());
                tokens.push("*".to_string());
                tokens.push(format_num_tok(val));
                tokens.push(")".to_string());
                i = next;
                continue;
            }
            if i + 1 < atoms.len() && is_digit_atom(&atoms[i + 1]) {
                tokens.push("(".to_string());
                tokens.push("2".to_string());
                tokens.push("*".to_string());
                tokens.push(atoms[i + 1].clone());
                tokens.push(")".to_string());
                i += 2;
                continue;
            }
            return None;
        }
        if w == "half" {
            // "half of <num>" -> "( <num> / 2 )".
            if i + 1 < atoms.len() && atoms[i + 1] == "of" {
                if let Some((val, next)) = parse_number_phrase(&atoms, i + 2) {
                    tokens.push("(".to_string());
                    tokens.push(format_num_tok(val));
                    tokens.push("/".to_string());
                    tokens.push("2".to_string());
                    tokens.push(")".to_string());
                    i = next;
                    continue;
                }
                if i + 2 < atoms.len() && is_digit_atom(&atoms[i + 2]) {
                    tokens.push("(".to_string());
                    tokens.push(atoms[i + 2].clone());
                    tokens.push("/".to_string());
                    tokens.push("2".to_string());
                    tokens.push(")".to_string());
                    i += 3;
                    continue;
                }
            }
            return None;
        }
        // "a dozen" / "a pair" as bare constants.
        if (w == "a" || w == "an")
            && i + 1 < atoms.len()
            && (atoms[i + 1] == "dozen" || atoms[i + 1] == "pair")
        {
            tokens.push(if atoms[i + 1] == "dozen" {
                "12".to_string()
            } else {
                "2".to_string()
            });
            i += 2;
            continue;
        }
        // Number-word phrases.
        if let Some((val, next)) = parse_number_phrase(&atoms, i) {
            tokens.push(format_num_tok(val));
            i = next;
            continue;
        }
        // Single-word operators.
        if let Some(sym) = word_operator(w) {
            tokens.push(sym.to_string());
            i += 1;
            continue;
        }
        // Suffix operators.
        if w == "squared" {
            tokens.push("^".to_string());
            tokens.push("2".to_string());
            i += 1;
            continue;
        }
        if w == "cubed" {
            tokens.push("^".to_string());
            tokens.push("3".to_string());
            i += 1;
            continue;
        }
        if w == "negative" {
            tokens.push("-".to_string());
            i += 1;
            continue;
        }
        // Digits pass through.
        if is_digit_atom(w) {
            tokens.push(w.to_string());
            i += 1;
            continue;
        }
        if ["+", "-", "*", "/", "%", "^", "(", ")"].contains(&w) {
            tokens.push(w.to_string());
            i += 1;
            continue;
        }
        // Whitelisted function identifiers and constants pass through;
        // everything else declines the prompt.
        if is_allowed_ident(w) || w == "pi" || w == "e" {
            tokens.push(w.to_string());
            i += 1;
            continue;
        }
        return None;
    }
    let expr: String = tokens.join(" ");
    // Must contain at least one arithmetic operator (not just a number).
    if !tokens
        .iter()
        .any(|t| ["+", "-", "*", "/", "%", "^"].contains(&t.as_str()))
    {
        return None;
    }
    Some(expr)
}

fn format_num_tok(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e15 {
        format!("{}", v.round() as i64)
    } else {
        format!("{}", v)
    }
}

fn is_allowed_ident(w: &str) -> bool {
    matches!(
        w,
        "sqrt" | "abs" | "round" | "floor" | "ceil" | "log" | "log10" | "exp" | "min" | "max" | "pow"
    )
}

// ── recursive-descent evaluator (calc.ts grammar, fail-closed) ────────────

#[derive(Debug)]
struct ParseError;

struct Parser {
    tokens: Vec<String>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }
    fn next(&mut self) -> Result<String, ParseError> {
        let t = self.tokens.get(self.pos).cloned().ok_or(ParseError)?;
        self.pos += 1;
        Ok(t)
    }
    fn parse_expr(&mut self) -> Result<f64, ParseError> {
        let mut v = self.parse_term()?;
        while matches!(self.peek(), Some("+") | Some("-")) {
            let op = self.next()?;
            let rhs = self.parse_term()?;
            v = if op == "+" { v + rhs } else { v - rhs };
        }
        Ok(v)
    }
    fn parse_term(&mut self) -> Result<f64, ParseError> {
        let mut v = self.parse_power()?;
        while matches!(self.peek(), Some("*") | Some("/") | Some("%")) {
            let op = self.next()?;
            let rhs = self.parse_power()?;
            v = match op.as_str() {
                "*" => v * rhs,
                "/" => {
                    if rhs == 0.0 {
                        return Err(ParseError);
                    }
                    v / rhs
                }
                _ => {
                    if rhs == 0.0 {
                        return Err(ParseError);
                    }
                    v % rhs
                }
            };
        }
        Ok(v)
    }
    fn parse_power(&mut self) -> Result<f64, ParseError> {
        let base = self.parse_unary()?;
        if self.peek() == Some("^") {
            self.next()?;
            return Ok(base.powf(self.parse_power()?));
        }
        Ok(base)
    }
    fn parse_unary(&mut self) -> Result<f64, ParseError> {
        match self.peek() {
            Some("-") => {
                self.next()?;
                Ok(-self.parse_unary()?)
            }
            Some("+") => {
                self.next()?;
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }
    fn parse_primary(&mut self) -> Result<f64, ParseError> {
        let t = self.next()?;
        if t == "(" {
            let v = self.parse_expr()?;
            if self.next()? != ")" {
                return Err(ParseError);
            }
            return Ok(v);
        }
        if t.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '.') {
            return t.parse::<f64>().map_err(|_| ParseError);
        }
        let low = t.to_lowercase();
        if low == "pi" {
            return Ok(std::f64::consts::PI);
        }
        if low == "e" {
            return Ok(std::f64::consts::E);
        }
        if is_allowed_ident(&low) {
            if self.next()? != "(" {
                return Err(ParseError);
            }
            let mut args = Vec::new();
            if self.peek() != Some(")") {
                args.push(self.parse_expr()?);
                while self.peek() == Some(",") {
                    self.next()?;
                    args.push(self.parse_expr()?);
                }
            }
            if self.next()? != ")" {
                return Err(ParseError);
            }
            return apply_func(&low, &args).ok_or(ParseError);
        }
        Err(ParseError)
    }
}

fn apply_func(name: &str, args: &[f64]) -> Option<f64> {
    let r = match name {
        "sqrt" => args.first()?.sqrt(),
        "abs" => args.first()?.abs(),
        "round" => args.first()?.round(),
        "floor" => args.first()?.floor(),
        "ceil" => args.first()?.ceil(),
        "log" => args.first()?.ln(),
        "log10" => args.first()?.log10(),
        "exp" => args.first()?.exp(),
        "min" => args.iter().cloned().reduce(f64::min)?,
        "max" => args.iter().cloned().reduce(f64::max)?,
        "pow" => {
            if args.len() != 2 {
                return None;
            }
            args[0].powf(args[1])
        }
        _ => return None,
    };
    if r.is_finite() { Some(r) } else { None }
}

/// Tokenize an expression string into tokens and evaluate fully.
/// Any trailing tokens or parse failure is an error.
fn eval_expression(expr: &str) -> Result<f64, ParseError> {
    let mut tokens: Vec<String> = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let mut s = String::new();
            while chars.peek().is_some_and(|n| n.is_ascii_digit() || *n == '.') {
                s.push(chars.next().unwrap());
            }
            tokens.push(s);
            continue;
        }
        if c.is_ascii_alphabetic() {
            let mut s = String::new();
            while chars.peek().is_some_and(|n| n.is_ascii_alphanumeric() || *n == '_') {
                s.push(chars.next().unwrap());
            }
            tokens.push(s);
            continue;
        }
        if "+-*/%^(),".contains(c) {
            tokens.push(c.to_string());
            chars.next();
            continue;
        }
        return Err(ParseError);
    }
    let mut p = Parser { tokens, pos: 0 };
    let v = p.parse_expr()?;
    if p.pos != p.tokens.len() {
        return Err(ParseError);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_constant_dozen() {
        let s = try_evaluate_deterministic_arithmetic("What is two dozen plus five").unwrap();
        assert_eq!(s.synthesized_response, "24 + 5 = 29.");
        let s = try_evaluate_deterministic_arithmetic("What is a dozen times three").unwrap();
        assert_eq!(s.synthesized_response, "12 * 3 = 36.");
    }

    #[test]
    fn implicit_constant_pair_and_weekly() {
        let s = try_evaluate_deterministic_arithmetic("What is a pair plus a pair").unwrap();
        assert_eq!(s.synthesized_response, "2 + 2 = 4.");
        let s = try_evaluate_deterministic_arithmetic("What is weekly times three").unwrap();
        assert_eq!(s.synthesized_response, "7 * 3 = 21.");
    }

    #[test]
    fn implicit_operator_twice_and_half() {
        let s = try_evaluate_deterministic_arithmetic("What is twice nine").unwrap();
        assert_eq!(s.synthesized_response, "( 2 * 9 ) = 18.");
        let s = try_evaluate_deterministic_arithmetic("What is seven twice plus one").unwrap();
        assert_eq!(s.synthesized_response, "7 * 2 + 1 = 15.");
        let s = try_evaluate_deterministic_arithmetic("What is half of forty eight").unwrap();
        assert_eq!(s.synthesized_response, "( 48 / 2 ) = 24.");
        let s = try_evaluate_deterministic_arithmetic("What is half of 48 plus 2").unwrap();
        assert_eq!(s.synthesized_response, "( 48 / 2 ) + 2 = 26.");
    }

    #[test]
    fn word_arithmetic_two_plus_two() {
        let s = try_evaluate_deterministic_arithmetic("What is two plus two").unwrap();
        assert_eq!(s.synthesized_response, "2 + 2 = 4.");
        assert_eq!(s.operator_family, "DeterministicArithmeticOperator");
    }

    #[test]
    fn digit_arithmetic_with_parens() {
        let s = try_evaluate_deterministic_arithmetic("3 * (4 + 5)").unwrap();
        assert_eq!(s.synthesized_response, "3 * ( 4 + 5 ) = 27.");
    }

    #[test]
    fn composite_number_words() {
        let s = try_evaluate_deterministic_arithmetic("one hundred and five minus five").unwrap();
        assert_eq!(s.synthesized_response, "105 - 5 = 100.");
    }

    #[test]
    fn division_by_zero_declines() {
        assert!(try_evaluate_deterministic_arithmetic("10 / 0").is_none());
    }

    #[test]
    fn non_math_declines() {
        assert!(try_evaluate_deterministic_arithmetic("two apples").is_none());
        assert!(try_evaluate_deterministic_arithmetic("two").is_none());
        assert!(try_evaluate_deterministic_arithmetic("tell me about paris").is_none());
        assert!(try_evaluate_deterministic_arithmetic("what is the capital of france").is_none());
    }

    #[test]
    fn subtracted_from_flips() {
        let s = try_evaluate_deterministic_arithmetic("five subtracted from ten").unwrap();
        assert_eq!(s.synthesized_response, "10 - 5 = 5.");
    }

    #[test]
    fn whitelisted_functions() {
        let s = try_evaluate_deterministic_arithmetic("sqrt(16) + abs(-3)").unwrap();
        assert!(s.synthesized_response.ends_with("= 7."));
    }

    #[test]
    fn word_problems_still_decline() {
        // Multi-clause word problems belong to the GSM8K path, not here.
        assert!(try_evaluate_deterministic_arithmetic(
            "John has 5 apples and gives 2 to Mary. How many does he have left?"
        )
        .is_none());
    }
}
