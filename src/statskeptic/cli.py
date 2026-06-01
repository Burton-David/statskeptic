"""Command-line surface. A thin adapter over `analyze`: it reads the CSV, formats the
report, and turns the verdict into an exit code so the tool is scriptable as a gate.

Exit codes follow the verdict, not just success/failure, because "the data cannot say"
is a distinct, useful signal from "here is a defensible answer" and from "I could not
even map your question".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import analyze
from .critique.models import Verdict
from .errors import AnalysisError
from .plan.models import PlanHints

# BSD sysexits: 64 is "you called me wrong", 70 is "I broke trying".
EXIT_USAGE = 64
EXIT_INTERNAL = 70

_VERDICT_EXIT = {
    Verdict.defensible: 0,
    Verdict.defensible_with_caveats: 0,
    Verdict.cannot_conclude: 2,
    Verdict.declined: 3,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="statskeptic",
        description="Analyze a dataset and red-team the conclusion.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze a CSV against a question")
    a.add_argument("data", type=Path, help="path to a CSV file")
    a.add_argument("-q", "--question", required=True, help="the question to answer")
    a.add_argument("--json", action="store_true", help="emit the report as JSON")
    a.add_argument("--outcome", help="name the outcome column (overrides matching)")
    a.add_argument("--group", "--by", dest="group", help="name the grouping column")
    a.add_argument(
        "--predictors", help="comma-separated predictor columns for regression"
    )
    a.add_argument("--alpha", type=float, default=0.05, help="significance level")
    a.add_argument(
        "--quiet", action="store_true", help="suppress the report body; exit code only"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(f"error: file not found: {args.data}", file=sys.stderr)
        return EXIT_USAGE

    predictors = args.predictors.split(",") if args.predictors else None
    hints = PlanHints(outcome=args.outcome, group=args.group, predictors=predictors)

    try:
        report = analyze(args.data, args.question, hints=hints, alpha=args.alpha)
    except AnalysisError as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not args.quiet:
        print(report.to_json() if args.json else report.explain())

    return _VERDICT_EXIT[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
