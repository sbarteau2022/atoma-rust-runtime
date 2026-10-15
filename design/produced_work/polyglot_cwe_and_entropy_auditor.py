#!/usr/bin/env python3
"""
Autonomous Monster-Hybrid Synthesized Tool: `polyglot_cwe_and_entropy_auditor.py`
Combines Superpowers Pillar 1 (Secret & CWE Scanner) with Pillar 4 (Secure Lifecycle Gate).
Zero external dependencies (Python 3 stdlib only).
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class SecurityFinding:
    cwe_id: str
    severity: str
    line_number: int
    description: str
    snippet: str
    entropy_bits: float


class PolyglotCweAuditor:
    PATTERNS: Dict[str, tuple[str, str, re.Pattern[str]]] = {
        "CWE-78": (
            "CRITICAL",
            "OS Command Injection via shell=True or unchecked system()",
            re.compile(r"(subprocess\.(?:Popen|run|call)\([^)]*shell\s*=\s*True|os\.system\()"),
        ),
        "CWE-89": (
            "HIGH",
            "Raw SQL string interpolation enabling SQL Injection",
            re.compile(r"(SELECT\s+.*\s+FROM\s+.*(%s|\{.*\}|\+\s*\w+))", re.IGNORECASE),
        ),
        "CWE-798": (
            "CRITICAL",
            "Hardcoded credential or high-entropy API secret literal",
            re.compile(r"(?:api_key|secret_key|private_key|token)\s*[:=]\s*['\"]([A-Za-z0-9_\-]{16,})['\"]", re.IGNORECASE),
        ),
    }

    @staticmethod
    def shannon_entropy(data: str) -> float:
        if not data:
            return 0.0
        freq: Dict[str, int] = {}
        for ch in data:
            freq[ch] = freq.get(ch, 0) + 1
        n = float(len(data))
        return -sum((c / n) * math.log2(c / n) for c in freq.values())

    def scan_source(self, source_code: str) -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        for idx, line in enumerate(source_code.splitlines(), start=1):
            for cwe_id, (severity, desc, regex) in self.PATTERNS.items():
                match = regex.search(line)
                if match:
                    token = match.group(1) if match.lastindex else match.group(0)
                    ent = self.shannon_entropy(token)
                    findings.append(
                        SecurityFinding(
                            cwe_id=cwe_id,
                            severity=severity,
                            line_number=idx,
                            description=desc,
                            snippet=line.strip(),
                            entropy_bits=round(ent, 3),
                        )
                    )
        return findings


def _run_self_test() -> None:
    auditor = PolyglotCweAuditor()
    sample = '''
    api_key = "X7kP9mQ2vR5tY8wN3bV6cX1zA4sD7fG0hJ"
    subprocess.run(cmd, shell=True)
    '''
    res = auditor.scan_source(sample)
    assert len(res) == 2, f"expected 2 findings, got {len(res)}"
    assert res[0].cwe_id == "CWE-798" and res[0].entropy_bits > 4.0
    assert res[1].cwe_id == "CWE-78"


if __name__ == "__main__":
    _run_self_test()
    print("[PASS] PolyglotCweAuditor self-test verified.")
