"""odeaudit CLI.

odeaudit list
odeaudit run [-p ROBER,HIRES,...] [-l scipy,odeint,diffrax,torchdiffeq] [--rtol 1e-3,1e-6,1e-9] [--timeout 60] [-o results]
odeaudit report results/results.json -o REPORT.md
"""

from __future__ import annotations

import argparse

from ._version import __version__
from .problems import PROBLEMS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="odeaudit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", action="version", version=f"odeaudit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list test problems")
    p = sub.add_parser("run", help="run solvers against references")
    p.add_argument("-p", "--problems", help="comma-separated subset")
    p.add_argument("-l", "--libraries", help="comma-separated subset of scipy,odeint,diffrax,torchdiffeq")
    p.add_argument("--rtol", default="1e-3,1e-6,1e-9")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("-o", "--out", default="results")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    r = sub.add_parser("report", help="write the Markdown report")
    r.add_argument("results")
    r.add_argument("-o", "--out", default="REPORT.md")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for pr in PROBLEMS.values():
            print(
                f"{pr.name:6s} {pr.group:13s} dim {pr.dim:2d} t_end {pr.t_span[1]:<8g} {'stiff' if pr.stiff else 'non-stiff':9s} {pr.description}"
            )
        return 0
    if args.cmd == "report":
        from .report import build_report

        build_report(args.results, args.out)
        print(f"wrote {args.out}")
        return 0
    from .run import run_audit

    problems = args.problems.split(",") if args.problems else None
    libraries = args.libraries.split(",") if args.libraries else None
    rtols = tuple(float(x) for x in args.rtol.split(","))
    run_audit(problems, libraries, rtols, args.timeout, args.out, resume=not args.no_resume, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
