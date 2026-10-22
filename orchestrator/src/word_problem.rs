//! Word-problem automaton — the MECHANICAL half of
//! `docs/MATH_WORKFLOW.md` (2026-10-20).
//!
//! This module validates, executes, and balances derivations. It does
//! not derive them. Stage 0/1 (a two-slot key and an equation, from the
//! words of the problem) lives in `word_problem_derive`. What lives here
//! is everything that must be true of a derivation before its answer may
//! be printed:
//!
//!   * the two-slot key (OperativeBindings / ContextualDistractors),
//!   * strictness: an equation may use operative bindings only; every
//!     operative binding must be used; distractor use is a refusal,
//!   * unit-tracked execution (Stage 2 delegation),
//!   * dimensional validation (Stage 3 Semantic Balancer): the
//!     answer's unit must equal the asked unit, or the derivation is
//!     refused as `dimensional_mismatch` — the False-Equilibrium trap
//!     (a wrong-structure equation whose numbers happen to balance)
//!     dies here, before any bet is filed.

use std::collections::BTreeMap;
use std::fmt;

// ── units ────────────────────────────────────────────────────────────────

/// Normalize a unit name to its singular dimension key ("muffins" ->
/// "muffin", "dollars" -> "dollar"). Deliberately small: trailing-s
/// stripping with the common exceptions.
pub fn singularize(word: &str) -> String {
    let w = word.trim().to_lowercase();
    let w = w.trim_matches('$');
    if w.len() > 3 && w.ends_with("ies") {
        return format!("{}y", &w[..w.len() - 3]);
    }
    if w.ends_with("sses") || w.ends_with("xes") || w.ends_with("ches") || w.ends_with("shes") {
        return w[..w.len() - 2].to_string();
    }
    if w.len() > 1 && w.ends_with('s') && !w.ends_with("ss") {
        return w[..w.len() - 1].to_string();
    }
    w.to_string()
}

/// A physical unit as dimension exponents over named base dimensions:
/// `muffin^1`, `meter^1 second^-1`, dimensionless = empty map.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Unit {
    dims: BTreeMap<String, i32>,
}

impl Unit {
    pub fn dimensionless() -> Self {
        Self::default()
    }

    pub fn of(name: &str) -> Self {
        let mut dims = BTreeMap::new();
        dims.insert(singularize(name), 1);
        Self { dims }
    }

    /// Parse "muffins", "meters per second", "dollars/hour".
    pub fn parse(text: &str) -> Self {
        let t = text.trim().to_lowercase();
        if t.is_empty() || t == "1" {
            return Self::dimensionless();
        }
        if let Some((num, den)) = t.split_once(" per ") {
            return Self::of(num).div(&Self::of(den));
        }
        if let Some((num, den)) = t.split_once('/') {
            return Self::of(num).div(&Self::of(den));
        }
        Self::of(&t)
    }

    pub fn mul(&self, other: &Unit) -> Unit {
        let mut dims = self.dims.clone();
        for (k, v) in &other.dims {
            let e = dims.entry(k.clone()).or_insert(0);
            *e += v;
            if *e == 0 {
                dims.remove(k);
            }
        }
        Unit { dims }
    }

    pub fn div(&self, other: &Unit) -> Unit {
        let mut dims = self.dims.clone();
        for (k, v) in &other.dims {
            let e = dims.entry(k.clone()).or_insert(0);
            *e -= v;
            if *e == 0 {
                dims.remove(k);
            }
        }
        Unit { dims }
    }

    pub fn pow(&self, n: i32) -> Unit {
        if n == 0 {
            return Self::dimensionless();
        }
        Unit {
            dims: self
                .dims
                .iter()
                .map(|(k, v)| (k.clone(), v * n))
                .filter(|(_, v)| *v != 0)
                .collect(),
        }
    }

    pub fn is_dimensionless(&self) -> bool {
        self.dims.is_empty()
    }

    /// Exponent of one base dimension. Unknown names are 0.
    pub fn exponent(&self, name: &str) -> i32 {
        self.dims.get(&singularize(name)).copied().unwrap_or(0)
    }

    /// Every dimension of `self` also occurs in `other`.
    pub fn dims_subset_of(&self, other: &Unit) -> bool {
        self.dims.keys().all(|k| other.dims.contains_key(k))
    }

    /// The two units share at least one base dimension.
    pub fn shares_dimension(&self, other: &Unit) -> bool {
        self.dims.keys().any(|k| other.dims.contains_key(k))
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            return write!(f, "1");
        }
        let mut parts: Vec<String> = Vec::new();
        for (k, v) in &self.dims {
            if *v == 1 {
                parts.push(k.clone());
            } else {
                parts.push(format!("{k}^{v}"));
            }
        }
        write!(f, "{}", parts.join(" "))
    }
}

// ── key (Stage 0) ────────────────────────────────────────────────────────

/// One symbol bound to a quantity: value + unit + the entity it names.
#[derive(Debug, Clone)]
pub struct Binding {
    pub symbol: String,
    pub value: f64,
    pub unit: Unit,
    pub label: String,
}

/// The two-slot key. Every quantity in the problem is classified into
/// exactly one slot; the classification itself is the deriver's claim,
/// judged later by outcomes.
#[derive(Debug, Clone)]
pub struct Key {
    pub operative: Vec<Binding>,
    pub distractors: Vec<Binding>,
    pub asked_unit: Unit,
}

impl Key {
    fn operative_binding(&self, symbol: &str) -> Option<&Binding> {
        self.operative.iter().find(|b| b.symbol == symbol)
    }
    fn is_distractor(&self, symbol: &str) -> bool {
        self.distractors.iter().any(|b| b.symbol == symbol)
    }
}

// ── equation AST (Stage 1 output form) ───────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(f64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Pow(Box<Expr>, i32),
}

impl Expr {
    fn vars(&self, out: &mut Vec<String>) {
        match self {
            Expr::Const(_) => {}
            Expr::Var(s) => out.push(s.clone()),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                a.vars(out);
                b.vars(out);
            }
            Expr::Neg(a) | Expr::Pow(a, _) => a.vars(out),
        }
    }

    fn eval(&self, key: &Key, trace: &mut Vec<String>) -> Result<(f64, Unit), Refusal> {
        match self {
            Expr::Const(v) => Ok((*v, Unit::dimensionless())),
            Expr::Var(s) => {
                let b = key
                    .operative_binding(s)
                    .ok_or_else(|| Refusal::UnknownSymbol(s.clone()))?;
                Ok((b.value, b.unit.clone()))
            }
            Expr::Add(a, b) => {
                let (x, ux) = a.eval(key, trace)?;
                let (y, uy) = b.eval(key, trace)?;
                if ux != uy {
                    return Err(Refusal::DimensionalMismatch {
                        detail: format!("cannot add {ux} and {uy}"),
                    });
                }
                trace.push(format!("{x} {ux} + {y} {uy} = {} {ux}", x + y));
                Ok((x + y, ux))
            }
            Expr::Sub(a, b) => {
                let (x, ux) = a.eval(key, trace)?;
                let (y, uy) = b.eval(key, trace)?;
                if ux != uy {
                    return Err(Refusal::DimensionalMismatch {
                        detail: format!("cannot subtract {uy} from {ux}"),
                    });
                }
                trace.push(format!("{x} {ux} - {y} {uy} = {} {ux}", x - y));
                Ok((x - y, ux))
            }
            Expr::Mul(a, b) => {
                let (x, ux) = a.eval(key, trace)?;
                let (y, uy) = b.eval(key, trace)?;
                let u = ux.mul(&uy);
                trace.push(format!("{x} {ux} * {y} {uy} = {} {u}", x * y));
                Ok((x * y, u))
            }
            Expr::Div(a, b) => {
                let (x, ux) = a.eval(key, trace)?;
                let (y, uy) = b.eval(key, trace)?;
                if y == 0.0 {
                    return Err(Refusal::DivisionByZero);
                }
                let u = ux.div(&uy);
                trace.push(format!("{x} {ux} / {y} {uy} = {} {u}", x / y));
                Ok((x / y, u))
            }
            Expr::Neg(a) => {
                let (x, ux) = a.eval(key, trace)?;
                Ok((-x, ux))
            }
            Expr::Pow(a, n) => {
                let (x, ux) = a.eval(key, trace)?;
                Ok((x.powi(*n), ux.pow(*n)))
            }
        }
    }
}

// ── refusals (named, diagnosable — never silent) ─────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    DimensionalMismatch { detail: String },
    DistractorBindingUsed(String),
    UnusedOperativeBinding(String),
    UnknownSymbol(String),
    DivisionByZero,
    ParseError(String),
}

impl Refusal {
    pub fn refusal_reason(&self) -> &'static str {
        match self {
            Refusal::DimensionalMismatch { .. } => "dimensional_mismatch",
            Refusal::DistractorBindingUsed(_) => "distractor_binding_used",
            Refusal::UnusedOperativeBinding(_) => "unused_operative_binding",
            Refusal::UnknownSymbol(_) => "unknown_symbol",
            Refusal::DivisionByZero => "division_by_zero",
            Refusal::ParseError(_) => "unparseable_keys",
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::DimensionalMismatch { detail } => {
                write!(f, "dimensional mismatch: {detail}")
            }
            Refusal::DistractorBindingUsed(s) => {
                write!(f, "equation uses distractor binding '{s}'")
            }
            Refusal::UnusedOperativeBinding(s) => {
                write!(f, "operative binding '{s}' never used (mis-parse)")
            }
            Refusal::UnknownSymbol(s) => write!(f, "unknown symbol '{s}'"),
            Refusal::DivisionByZero => write!(f, "division by zero"),
            Refusal::ParseError(e) => write!(f, "parse error: {e}"),
        }
    }
}

// ── validation + solve (Stages 0-strictness, 2, 3) ─────────────────────

#[derive(Debug, Clone)]
pub struct Solution {
    pub value: f64,
    pub unit: Unit,
    pub trace: Vec<String>,
}

/// Validate a derivation against its key and execute it with unit
/// tracking. Succeeds only when: every referenced symbol is an
/// operative binding, every operative binding is used, all operations
/// are dimensionally sound, and the result unit equals the asked unit.
pub fn validate_and_solve(key: &Key, equation: &Expr) -> Result<Solution, Refusal> {
    let mut used: Vec<String> = Vec::new();
    equation.vars(&mut used);
    for s in &used {
        if key.is_distractor(s) {
            return Err(Refusal::DistractorBindingUsed(s.clone()));
        }
        if key.operative_binding(s).is_none() {
            return Err(Refusal::UnknownSymbol(s.clone()));
        }
    }
    for b in &key.operative {
        if !used.iter().any(|u| u == &b.symbol) {
            return Err(Refusal::UnusedOperativeBinding(b.symbol.clone()));
        }
    }
    let mut trace = Vec::new();
    let (value, unit) = equation.eval(key, &mut trace)?;
    if unit != key.asked_unit {
        return Err(Refusal::DimensionalMismatch {
            detail: format!(
                "answer unit {unit} does not match asked unit {}",
                key.asked_unit
            ),
        });
    }
    Ok(Solution { value, unit, trace })
}

/// Stage 3 answer confirmation: the answer exists only if evaluating
/// the original equation yields the claimed value AND the asked unit.
pub fn confirm_answer(
    key: &Key,
    equation: &Expr,
    claimed_value: f64,
) -> Result<bool, Refusal> {
    let sol = validate_and_solve(key, equation)?;
    Ok((sol.value - claimed_value).abs() < 1e-9 && sol.unit == key.asked_unit)
}

/// Stage 3 balance check: both sides of an asserted equation evaluate
/// to the same value AND the same unit.
pub fn check_balance(key: &Key, lhs: &Expr, rhs: &Expr) -> Result<bool, Refusal> {
    let mut t = Vec::new();
    let (x, ux) = lhs.eval(key, &mut t)?;
    let (y, uy) = rhs.eval(key, &mut t)?;
    if ux != uy {
        return Err(Refusal::DimensionalMismatch {
            detail: format!("balance sides differ in unit: {ux} vs {uy}"),
        });
    }
    Ok((x - y).abs() < 1e-9)
}

// ── expression parser (verbose-math surface syntax) ────────────────────
//
// Grammar: expr := term (('+'|'-') term)* ; term := factor (('*'|'/')
// factor)* ; factor := unary ('^' int)? ; unary := '-' unary | atom ;
// atom := number | ident | '(' expr ')'.

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

fn lex(input: &str) -> Result<Vec<Tok>, Refusal> {
    let mut toks = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' => i += 1,
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                toks.push(Tok::Star);
                i += 1;
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                toks.push(Tok::Caret);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            '0'..='9' | '.' => {
                let mut j = i;
                while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                    j += 1;
                }
                let s: String = chars[i..j].iter().collect();
                let v: f64 = s
                    .parse()
                    .map_err(|_| Refusal::ParseError(format!("bad number '{s}'")))?;
                toks.push(Tok::Num(v));
                i = j;
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut j = i;
                while j < chars.len()
                    && (chars[j].is_ascii_alphanumeric() || chars[j] == '_')
                {
                    j += 1;
                }
                toks.push(Tok::Ident(chars[i..j].iter().collect()));
                i = j;
            }
            _ => return Err(Refusal::ParseError(format!("unexpected char '{c}'"))),
        }
    }
    Ok(toks)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn expr(&mut self) -> Result<Expr, Refusal> {
        let mut lhs = self.term()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.next();
                    let rhs = self.term()?;
                    lhs = Expr::Add(Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Minus) => {
                    self.next();
                    let rhs = self.term()?;
                    lhs = Expr::Sub(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }
    fn term(&mut self) -> Result<Expr, Refusal> {
        let mut lhs = self.factor()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.next();
                    let rhs = self.factor()?;
                    lhs = Expr::Mul(Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Slash) => {
                    self.next();
                    let rhs = self.factor()?;
                    lhs = Expr::Div(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }
    fn factor(&mut self) -> Result<Expr, Refusal> {
        let base = self.unary()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.next();
            match self.next() {
                Some(Tok::Num(v)) if v.fract() == 0.0 => {
                    return Ok(Expr::Pow(Box::new(base), v as i32));
                }
                _ => return Err(Refusal::ParseError("exponent must be an integer".into())),
            }
        }
        Ok(base)
    }
    fn unary(&mut self) -> Result<Expr, Refusal> {
        if matches!(self.peek(), Some(Tok::Minus)) {
            self.next();
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.atom()
    }
    fn atom(&mut self) -> Result<Expr, Refusal> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Expr::Const(v)),
            Some(Tok::Ident(s)) => Ok(Expr::Var(s)),
            Some(Tok::LParen) => {
                let e = self.expr()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err(Refusal::ParseError("missing ')'".into())),
                }
            }
            other => Err(Refusal::ParseError(format!(
                "unexpected token {other:?}"
            ))),
        }
    }
}

/// Parse a verbose-math equation string into an `Expr`.
pub fn parse_expression(input: &str) -> Result<Expr, Refusal> {
    let toks = lex(input)?;
    let mut p = Parser { toks, pos: 0 };
    let e = p.expr()?;
    if p.pos != p.toks.len() {
        return Err(Refusal::ParseError("trailing tokens".into()));
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baker_key() -> Key {
        Key {
            operative: vec![
                Binding {
                    symbol: "trays".into(),
                    value: 6.0,
                    unit: Unit::of("trays"),
                    label: "trays of muffins".into(),
                },
                Binding {
                    symbol: "per_tray".into(),
                    value: 12.0,
                    unit: Unit::parse("muffins per tray"),
                    label: "muffins on each tray".into(),
                },
                Binding {
                    symbol: "sold".into(),
                    value: 49.0,
                    unit: Unit::of("muffins"),
                    label: "muffins sold in the morning".into(),
                },
            ],
            distractors: vec![Binding {
                symbol: "bakers_age".into(),
                value: 41.0,
                unit: Unit::of("years"),
                label: "the baker's age".into(),
            }],
            asked_unit: Unit::of("muffins"),
        }
    }

    #[test]
    fn baker_problem_solves_with_units() {
        let key = baker_key();
        let eq = parse_expression("trays * per_tray - sold").unwrap();
        let sol = validate_and_solve(&key, &eq).unwrap();
        assert_eq!(sol.value, 23.0);
        assert_eq!(sol.unit, Unit::of("muffins"));
    }

    #[test]
    fn false_equilibrium_dimensional_trap_refused() {
        // 5 apples * 3 apples = 15 apples^2: numbers fine, structure
        // wrong. Asked unit is apples -> dimensional refusal.
        let key = Key {
            operative: vec![
                Binding {
                    symbol: "a".into(),
                    value: 5.0,
                    unit: Unit::of("apples"),
                    label: "apples in basket".into(),
                },
                Binding {
                    symbol: "b".into(),
                    value: 3.0,
                    unit: Unit::of("apples"),
                    label: "more apples".into(),
                },
            ],
            distractors: vec![],
            asked_unit: Unit::of("apples"),
        };
        let eq = parse_expression("a * b").unwrap();
        let err = validate_and_solve(&key, &eq).unwrap_err();
        assert_eq!(err.refusal_reason(), "dimensional_mismatch");
        // The correct structure passes.
        let eq = parse_expression("a + b").unwrap();
        let sol = validate_and_solve(&key, &eq).unwrap();
        assert_eq!(sol.value, 8.0);
    }

    #[test]
    fn distractor_use_and_unused_operative_refused() {
        let key = baker_key();
        let eq = parse_expression("trays * per_tray - bakers_age").unwrap();
        let err = validate_and_solve(&key, &eq).unwrap_err();
        assert_eq!(err.refusal_reason(), "distractor_binding_used");
        let eq = parse_expression("trays * per_tray").unwrap();
        let err = validate_and_solve(&key, &eq).unwrap_err();
        assert_eq!(err.refusal_reason(), "unused_operative_binding");
    }

    #[test]
    fn unit_algebra_and_balance() {
        let rate = Unit::parse("meters per second");
        let time = Unit::of("seconds");
        assert_eq!(rate.mul(&time), Unit::of("meters"));
        assert_eq!(Unit::of("muffins").pow(2).to_string(), "muffin^2");
        let key = baker_key();
        let lhs = parse_expression("trays * per_tray - sold").unwrap();
        assert!(confirm_answer(&key, &lhs, 23.0).unwrap());
        assert!(!confirm_answer(&key, &lhs, 121.0).unwrap());
    }
}
