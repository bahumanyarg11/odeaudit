"""Solver adapters. Each returns the solution at the requested times plus bookkeeping.

All adapters take ``(problem, times, rtol, atol)`` and return a :class:`Run`. They are invoked by name
inside a child process (see :mod:`odeaudit.run`) so a hung solver can be killed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .problems import Problem, rhs_for_backend

SCIPY_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")
DIFFRAX_SOLVERS = ("Tsit5", "Dopri5", "Dopri8", "Kvaerno5")
TORCHDIFFEQ_METHODS = ("dopri5", "dopri8", "bosh3", "adaptive_heun")


@dataclass
class Run:
    solver: str
    success: bool
    values: list[list[float]] = field(default_factory=list)  # (n_times, dim)
    nfev: int | None = None
    nsteps: int | None = None
    seconds: float = 0.0
    message: str = ""


def _scipy(method: str) -> Callable:
    from scipy.integrate import solve_ivp

    def solve(p: Problem, times: list[float], rtol: float, atol: float) -> Run:
        f = rhs_for_backend(p, np)
        kw = {}
        if method in ("Radau", "BDF", "LSODA") and p.jac is not None:
            kw["jac"] = p.jac
        t0 = time.perf_counter()
        sol = solve_ivp(f, p.t_span, np.asarray(p.y0, float), method=method, rtol=rtol, atol=atol, t_eval=times, **kw)
        dt = time.perf_counter() - t0
        return Run(
            f"scipy.{method}",
            bool(sol.success),
            sol.y.T.tolist() if sol.success else [],
            int(sol.nfev),
            int(sol.t.size),
            dt,
            str(sol.message),
        )

    return solve


def _odeint(p: Problem, times: list[float], rtol: float, atol: float) -> Run:
    from scipy.integrate import odeint

    f = rhs_for_backend(p, np)
    t0 = time.perf_counter()
    ts = [p.t_span[0], *times]
    y, info = odeint(
        f, np.asarray(p.y0, float), ts, rtol=rtol, atol=atol, tfirst=True, full_output=True, mxstep=5_000_000
    )
    dt = time.perf_counter() - t0
    ok = info["message"] == "Integration successful."
    return Run(
        "scipy.odeint",
        ok,
        y[1:].tolist() if ok else [],
        int(info["nfe"][-1]),
        int(info["nst"][-1]),
        dt,
        info["message"],
    )


def _diffrax(name: str) -> Callable:
    def solve(p: Problem, times: list[float], rtol: float, atol: float) -> Run:
        import diffrax
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        f = rhs_for_backend(p, jnp)
        term = diffrax.ODETerm(lambda t, y, args: f(t, y))
        solver = getattr(diffrax, name)()
        ctrl = diffrax.PIDController(rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=jnp.asarray(times))
        t0 = time.perf_counter()
        try:
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=p.t_span[0],
                t1=p.t_span[1],
                dt0=None,
                y0=jnp.asarray(p.y0, dtype=jnp.float64),
                saveat=saveat,
                stepsize_controller=ctrl,
                max_steps=4_000_000,
                throw=False,
            )
        except Exception as exc:  # noqa: BLE001
            return Run(
                f"diffrax.{name}", False, [], None, None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"[:200]
            )
        dt = time.perf_counter() - t0
        ok = bool((sol.result == diffrax.RESULTS.successful).all()) if hasattr(sol, "result") else True
        vals = np.asarray(sol.ys, float)
        ok = ok and bool(np.all(np.isfinite(vals)))
        return Run(
            f"diffrax.{name}",
            ok,
            vals.tolist() if ok else [],
            int(sol.stats.get("num_steps", 0)) if hasattr(sol, "stats") else None,
            int(sol.stats.get("num_accepted_steps", 0)) if hasattr(sol, "stats") else None,
            dt,
            str(sol.result) if hasattr(sol, "result") else "",
        )

    return solve


def _torchdiffeq(method: str) -> Callable:
    def solve(p: Problem, times: list[float], rtol: float, atol: float) -> Run:
        import torch
        from torchdiffeq import odeint

        f = rhs_for_backend(p, torch)
        y0 = torch.tensor(p.y0, dtype=torch.float64)
        ts = torch.tensor([p.t_span[0], *times], dtype=torch.float64)
        nfev = [0]

        def func(t, y):
            nfev[0] += 1
            return f(t, y)

        t0 = time.perf_counter()
        try:
            ys = odeint(func, y0, ts, rtol=rtol, atol=atol, method=method)
        except Exception as exc:  # noqa: BLE001
            return Run(
                f"torchdiffeq.{method}",
                False,
                [],
                nfev[0],
                None,
                time.perf_counter() - t0,
                f"{type(exc).__name__}: {exc}"[:200],
            )
        dt = time.perf_counter() - t0
        vals = ys[1:].detach().numpy()
        ok = bool(np.all(np.isfinite(vals)))
        return Run(
            f"torchdiffeq.{method}",
            ok,
            vals.tolist() if ok else [],
            nfev[0],
            None,
            dt,
            "" if ok else "non-finite values",
        )

    return solve


def available_solvers(libraries: list[str] | None = None) -> dict[str, Callable]:
    libraries = libraries or ["scipy", "odeint", "diffrax", "torchdiffeq"]
    out: dict[str, Callable] = {}
    if "scipy" in libraries:
        for m in SCIPY_METHODS:
            out[f"scipy.{m}"] = _scipy(m)
    if "odeint" in libraries:
        out["scipy.odeint"] = _odeint
    if "diffrax" in libraries:
        try:
            import diffrax  # noqa: F401

            for s in DIFFRAX_SOLVERS:
                out[f"diffrax.{s}"] = _diffrax(s)
        except ImportError:
            pass
    if "torchdiffeq" in libraries:
        try:
            import torchdiffeq  # noqa: F401

            for m in TORCHDIFFEQ_METHODS:
                out[f"torchdiffeq.{m}"] = _torchdiffeq(m)
        except ImportError:
            pass
    return out


def solver_by_name(name: str) -> Callable:
    lib, _, method = name.partition(".")
    if lib == "scipy" and method == "odeint":
        return _odeint
    if lib == "scipy":
        return _scipy(method)
    if lib == "diffrax":
        return _diffrax(method)
    if lib == "torchdiffeq":
        return _torchdiffeq(method)
    raise KeyError(name)


def library_versions() -> dict[str, str]:
    out = {}
    import scipy

    out["scipy"] = scipy.__version__
    for mod in ("diffrax", "jax", "torchdiffeq", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "not installed"
    return out
