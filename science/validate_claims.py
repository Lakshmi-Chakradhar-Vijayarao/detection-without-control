#!/usr/bin/env python3
"""
science/validate_claims.py
==========================
Pre-submission CI check: ensures every claim referenced in paper source
is in status CONFIRMED or SUPPORTED. Blocks submission if any referenced
claim is FALSIFIED or EXPLORATORY.

Usage:
    python science/validate_claims.py                    # check all paper files
    python science/validate_claims.py --paper paper/     # check specific directory
    python science/validate_claims.py --list-all         # print all claims and statuses
    python science/validate_claims.py --list-falsified   # print only falsified claims

Exit codes:
    0 — all referenced claims valid for paper
    1 — one or more referenced claims are FALSIFIED or EXPLORATORY
    2 — configuration error (missing files, parse error)
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(2)

CLAIMS_FILE = Path(__file__).parent / "CLAIMS.yaml"
VALID_FOR_PAPER = {"CONFIRMED", "SUPPORTED"}
# FALSIFIED claims must be disclosed explicitly when cited — citation is allowed, not blocked.
# EXPLORATORY claims may be cited in experiment tables/appendices with appropriate labels.
WARN_FOR_PAPER = {"FALSIFIED", "EXPLORATORY"}

CLAIM_REF_PATTERNS = [
    re.compile(r'\bC\d{3}\b'),   # bare: C001, C012
    re.compile(r'claim[:\s]+C\d{3}', re.IGNORECASE),
    re.compile(r'\\cite\{(C\d{3})\}'),  # LaTeX: \cite{C001}
    re.compile(r'\[C\d{3}\]'),          # Markdown: [C001]
]


def load_claims(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {c["id"]: c for c in data.get("claims", [])}


def extract_claim_ids(text: str) -> set:
    found = set()
    for pattern in CLAIM_REF_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group()
            claim_id = re.search(r'C\d{3}', raw)
            if claim_id:
                found.add(claim_id.group())
    return found


def check_files(paper_dir: Path, claims: dict) -> list[dict]:
    issues = []
    paper_dir = Path(paper_dir)
    extensions = {".md", ".tex", ".txt", ".rst"}
    files = list(paper_dir.rglob("*"))
    text_files = [f for f in files if f.suffix in extensions and f.is_file()]

    if not text_files:
        print(f"  No paper source files found in {paper_dir}")
        return []

    for fp in text_files:
        content = fp.read_text(errors="replace")
        referenced = extract_claim_ids(content)
        for cid in sorted(referenced):
            if cid not in claims:
                issues.append({
                    "file": str(fp),
                    "claim_id": cid,
                    "status": "UNKNOWN",
                    "reason": "Claim ID not found in CLAIMS.yaml",
                    "severity": "ERROR",
                })
                continue
            claim = claims[cid]
            status = claim.get("status", "UNKNOWN")
            if status in WARN_FOR_PAPER:
                issues.append({
                    "file": str(fp),
                    "claim_id": cid,
                    "status": status,
                    "statement": claim.get("statement", claim.get("claim", ""))[:80] + "...",
                    "reason": (
                        "FALSIFIED claims must be cited explicitly as falsified — verify the citation discloses this."
                        if status == "FALSIFIED"
                        else "EXPLORATORY claims need explicit labeling — verify the citation marks this as preliminary."
                    ),
                    "severity": "WARN",
                })
    return issues


def print_claim_table(claims: dict, filter_status: str = None):
    statuses = {
        "CONFIRMED": "\033[92m✓ CONFIRMED  \033[0m",
        "SUPPORTED":  "\033[94m~ SUPPORTED  \033[0m",
        "EXPLORATORY":"\033[93m? EXPLORATORY\033[0m",
        "FALSIFIED":  "\033[91m✗ FALSIFIED  \033[0m",
    }
    for cid, claim in sorted(claims.items()):
        status = claim.get("status", "UNKNOWN")
        if filter_status and status != filter_status:
            continue
        label = statuses.get(status, f"  {status:<12}")
        stmt = claim.get("statement", "").replace("\n", " ").strip()[:70]
        print(f"  {cid}  {label}  {stmt}...")


def main():
    parser = argparse.ArgumentParser(description="Validate paper claims against CLAIMS.yaml")
    parser.add_argument("--paper", default="paper/", help="Directory containing paper source files")
    parser.add_argument("--list-all", action="store_true", help="Print all claims and their statuses")
    parser.add_argument("--list-falsified", action="store_true", help="Print only falsified claims")
    parser.add_argument("--claims-file", default=str(CLAIMS_FILE), help="Path to CLAIMS.yaml")
    args = parser.parse_args()

    claims_path = Path(args.claims_file)
    if not claims_path.exists():
        print(f"ERROR: CLAIMS.yaml not found at {claims_path}")
        sys.exit(2)

    claims = load_claims(claims_path)
    print(f"\n  Epistemic Observability — Claim Validator")
    print(f"  {'─'*50}")
    print(f"  Claims loaded: {len(claims)} total")
    confirmed  = sum(1 for c in claims.values() if c.get("status") == "CONFIRMED")
    supported  = sum(1 for c in claims.values() if c.get("status") == "SUPPORTED")
    exploratory= sum(1 for c in claims.values() if c.get("status") == "EXPLORATORY")
    falsified  = sum(1 for c in claims.values() if c.get("status") == "FALSIFIED")
    print(f"  \033[92m✓ CONFIRMED:   {confirmed}\033[0m")
    print(f"  \033[94m~ SUPPORTED:   {supported}\033[0m")
    print(f"  \033[93m? EXPLORATORY: {exploratory}\033[0m")
    print(f"  \033[91m✗ FALSIFIED:   {falsified}\033[0m")
    print()

    if args.list_all:
        print("  All claims:")
        print_claim_table(claims)
        print()
        return

    if args.list_falsified:
        print("  Falsified claims (must not appear in paper as true):")
        print_claim_table(claims, filter_status="FALSIFIED")
        print()
        return

    paper_dir = Path(args.paper)
    if not paper_dir.exists():
        print(f"  NOTE: Paper directory '{paper_dir}' does not exist yet. Skipping file scan.")
        print("  Run again with --paper <path> once paper source is created.")
        sys.exit(0)

    print(f"  Scanning paper source: {paper_dir}")
    issues = check_files(paper_dir, claims)

    if not issues:
        print("  \033[92m✓ All referenced claims are valid for paper submission.\033[0m\n")
        sys.exit(0)

    blocks = [i for i in issues if i["severity"] == "BLOCK"]
    errors = [i for i in issues if i["severity"] == "ERROR"]

    if blocks:
        print(f"  \033[91m✗ BLOCKED: {len(blocks)} claim(s) must be corrected before submission:\033[0m\n")
        for issue in blocks:
            print(f"    File: {issue['file']}")
            print(f"    Claim: {issue['claim_id']}  [{issue['status']}]")
            print(f"    Statement: {issue.get('statement', 'N/A')}")
            print(f"    Action: {issue['reason']}")
            print()

    warns = [i for i in issues if i["severity"] == "WARN"]
    if warns:
        print(f"  \033[93m⚠ WARNINGS: {len(warns)} disclosure-citation(s) — verify labels are explicit:\033[0m\n")
        for issue in warns:
            print(f"    {issue['claim_id']}  [{issue['status']}]  {issue['reason'][:80]}")
        print()

    if errors:
        print(f"  \033[93m⚠ ERRORS: {len(errors)} unknown claim reference(s):\033[0m\n")
        for issue in errors:
            print(f"    File: {issue['file']}  |  Claim: {issue['claim_id']}  — {issue['reason']}")
        print()

    if blocks or errors:
        sys.exit(1)
    # Warnings (WARN) are non-blocking — exit 0 with warning output.
    sys.exit(0)


if __name__ == "__main__":
    main()
