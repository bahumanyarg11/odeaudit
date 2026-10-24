"""Reference solutions at the checkpoints of each problem.

Three sources, recorded with every reference so a reader knows how far to trust it:

* ``analytic``  closed form (DETEST A1–A4, D1–D5 via Kepler's equation): ~1e-15 relative.
* ``taylor``    mpmath ``odefun`` (arbitrary-precision Taylor series) at 25 digits for the other
                non-stiff problems: ~1e-18 or better.
* ``consensus`` for stiff problems, where a Taylor integrator is hopeless: SciPy Radau at
                rtol = 1e-12 as the value, with the disagreement against BDF at the same tolerance
                reported as ``uncertainty`` (typically 1e-9 .. 1e-11). Achieved errors below that
                level cannot be resolved and are reported as "≤ reference uncertainty".
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from .problems import PROBLEMS, Problem, rhs_for_backend

TAYLOR_DPS = 25


@dataclass
class Reference:
    problem: str
    times: list[float]  # checkpoints followed by t_end
    values: list[list[float]]  # (n_times, dim)
    source: str  # analytic | taylor | consensus
    uncertainty: float  # estimated relative error of the reference itself
    seconds: float = 0.0
    note: str = ""


def checkpoint_times(p: Problem) -> list[float]:
    return [*p.checkpoints, p.t_span[1]]


class _MP:
    """Minimal array-module shim so the numpy-style right-hand sides run on lists of mpf."""

    @staticmethod
    def stack(parts):
        return list(parts)

    @staticmethod
    def asarray(x):
        return list(x)

    sqrt = staticmethod(mp.sqrt)
    cos = staticmethod(mp.cos)
    sin = staticmethod(mp.sin)
    exp = staticmethod(mp.exp)
    abs = staticmethod(abs)


def _taylor(p: Problem, times: list[float]) -> tuple[list[list[float]], float]:
    f = rhs_for_backend(p, _MP)
    with mp.workdps(TAYLOR_DPS):
        y0 = [mp.mpf(v) for v in p.y0]
        if p.dim == 1:
            sol = mp.odefun(lambda t, y: f(t, y), mp.mpf(p.t_span[0]), y0[0], tol=mp.mpf(10) ** (-TAYLOR_DPS + 3))
            vals = [[float(sol(mp.mpf(t)))] for t in times]
        else:
            sol = mp.odefun(lambda t, y: f(t, y), mp.mpf(p.t_span[0]), y0, tol=mp.mpf(10) ** (-TAYLOR_DPS + 3))
            vals = [[float(v) for v in sol(mp.mpf(t))] for t in times]
    return vals, 10.0 ** (-TAYLOR_DPS + 6)


def _consensus(p: Problem, times: list[float]) -> tuple[list[list[float]], float, str]:
    f = rhs_for_backend(p, np)
    atol = 1e-14 * p.scale
    kw = dict(rtol=1e-12, atol=atol, t_eval=times, dense_output=False)
    radau = solve_ivp(f, p.t_span, np.asarray(p.y0, float), method="Radau", jac=p.jac, **kw)
    bdf = solve_ivp(f, p.t_span, np.asarray(p.y0, float), method="BDF", jac=p.jac, **kw)
    if not radau.success:
        raise RuntimeError(f"Radau reference failed for {p.name}: {radau.message}")
    vals = radau.y.T
    diff = None
    if bdf.success and bdf.y.shape == radau.y.shape:
        denom = np.maximum(np.abs(vals), 1e-6 * p.scale)
        diff = np.abs(vals - bdf.y.T) / denom
        diff = diff[np.isfinite(diff)]
    if diff is not None and diff.size:
        unc = float(diff.max())
        note = "Radau rtol=1e-12; uncertainty = max relative disagreement with BDF rtol=1e-12"
    else:
        unc = 1e-8
        note = "Radau rtol=1e-12; BDF cross-check unavailable, uncertainty assumed 1e-8"
    return vals.tolist(), unc, note


def compute_reference(p: Problem) -> Reference:
    times = checkpoint_times(p)
    t0 = time.perf_counter()
    if p.exact is not None:
        vals = [np.asarray(p.exact(t), float).tolist() for t in times]
        ref = Reference(p.name, times, vals, "analytic", 1e-15)
    elif not p.stiff:
        vals, unc = _taylor(p, times)
        ref = Reference(p.name, times, vals, "taylor", unc)
    else:
        vals, unc, note = _consensus(p, times)
        ref = Reference(p.name, times, vals, "consensus", unc, note=note)
    ref.seconds = time.perf_counter() - t0
    return ref


def load_or_compute(names: list[str], cache: str | Path | None, verbose: bool = True) -> dict[str, Reference]:
    refs: dict[str, Reference] = {}
    cache_path = Path(cache) if cache else None
    if cache_path and cache_path.exists():
        for d in json.loads(cache_path.read_text()):
            refs[d["problem"]] = Reference(**d)
    for name in names:
        if name in refs:
            continue
        p = PROBLEMS[name]
        try:
            refs[name] = compute_reference(p)
        except Exception as exc:  # noqa: BLE001 - keep going, record the failure
            refs[name] = Reference(
                name, checkpoint_times(p), [], "failed", math.inf, note=f"{type(exc).__name__}: {exc}"
            )
        if verbose:
            r = refs[name]
            print(
                f"reference {name:6s} {r.source:9s} unc {r.uncertainty:.1e}  ({r.seconds:.1f} s) {r.note}", flush=True
            )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps([asdict(r) for r in refs.values()], indent=1))
    return refs
