"""Runner: every (problem, solver, rtol) in its own child process with a wall-clock limit.

A solver that does not finish a stiff problem within the budget is recorded as ``timeout``: for an
explicit method on ROBER that is the expected (and important) outcome, not an error of the harness.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ._version import __version__
from .metrics import assess
from .problems import PROBLEMS
from .references import Reference, checkpoint_times, load_or_compute
from .solvers import available_solvers, library_versions, solver_by_name

DEFAULT_RTOLS = (1e-3, 1e-6, 1e-9)


@dataclass
class Case:
    problem: str
    solver: str
    rtol: float
    atol: float
    status: str  # ok | failed | timeout | crashed
    seconds: float = 0.0
    nfev: int | None = None
    nsteps: int | None = None
    message: str = ""
    assessment: dict = field(default_factory=dict)


def _child(name: str, solver: str, rtol: float, atol: float, q) -> None:
    try:
        p = PROBLEMS[name]
        run = solver_by_name(solver)(p, checkpoint_times(p), rtol, atol)
        q.put(asdict(run))
    except Exception as exc:  # noqa: BLE001
        q.put({"crash": f"{type(exc).__name__}: {str(exc)[:300]}"})


def run_case(name: str, solver: str, rtol: float, atol: float, timeout: float, ref: Reference) -> Case:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(name, solver, rtol, atol, q), daemon=True)
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return Case(name, solver, rtol, atol, "timeout", timeout, message=f"no result within {timeout:.0f} s")
    dt = time.perf_counter() - t0
    try:
        res = q.get(timeout=5)
    except Exception:  # noqa: BLE001
        return Case(
            name, solver, rtol, atol, "crashed", dt, message=f"child exited (code {proc.exitcode}) without a result"
        )
    if "crash" in res:
        return Case(name, solver, rtol, atol, "crashed", dt, message=res["crash"])
    case = Case(
        name,
        solver,
        rtol,
        atol,
        "ok" if res["success"] else "failed",
        res["seconds"],
        res["nfev"],
        res["nsteps"],
        res["message"],
    )
    if res["success"] and ref.values:
        case.assessment = assess(
            np.asarray(res["values"], float),
            np.asarray(ref.values, float),
            rtol,
            atol,
            PROBLEMS[name].scale,
            ref.uncertainty,
        )
    return case


def run_audit(
    problems: list[str] | None = None,
    libraries: list[str] | None = None,
    rtols: tuple[float, ...] = DEFAULT_RTOLS,
    timeout: float = 60.0,
    out_dir: str | Path = "results",
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = problems or list(PROBLEMS)
    solvers = available_solvers(libraries)
    refs = load_or_compute(problems, out_dir / "references.json", verbose)
    results_path = out_dir / "results.json"
    cases: list[Case] = []
    if resume and results_path.exists():
        cases = [Case(**c) for c in json.loads(results_path.read_text())["cases"]]
    done = {(c.problem, c.solver, c.rtol) for c in cases}
    meta = {
        "odeaudit": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions(),
        "rtols": list(rtols),
        "timeout_s": timeout,
        "atol_rule": "atol = rtol * 1e-3 * problem.scale",
    }
    for name in problems:
        p = PROBLEMS[name]
        ref = refs[name]
        if not ref.values:
            if verbose:
                print(f"skip {name}: no reference ({ref.note})")
            continue
        for solver in solvers:
            for rtol in rtols:
                if (name, solver, rtol) in done:
                    continue
                atol = rtol * 1e-3 * p.scale
                case = run_case(name, solver, rtol, atol, timeout, ref)
                cases.append(case)
                if verbose:
                    a = case.assessment
                    tail = (
                        f"scd {a['scd']:5.2f}  err/rtol {a['tol_ratio']:9.3g}  {a['verdict']}"
                        if a
                        else f"{case.status}: {case.message[:60]}"
                    )
                    print(
                        f"{name:6s} {solver:22s} rtol {rtol:.0e}  {case.seconds:6.2f}s  nfev {str(case.nfev):>8s}  {tail}",
                        flush=True,
                    )
                results_path.write_text(json.dumps({"meta": meta, "cases": [asdict(c) for c in cases]}, indent=1))
    payload = {"meta": meta, "cases": [asdict(c) for c in cases]}
    results_path.write_text(json.dumps(payload, indent=1))
    return payload
