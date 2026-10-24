"""Test problems: the DETEST non-stiff set (Hull, Enright, Fellen & Sedgwick 1972; Enright & Pryce 1987)
and the classic stiff problems of Hairer & Wanner (ROBER, VDPOL, HIRES, OREGO, Chemical Akzo Nobel, E5).

Each problem gives ``f(t, y)``, optionally the Jacobian, ``y0``, ``t_span`` and, where one exists, an
analytic solution. For the others the reference is computed by :mod:`odeaudit.references`.
Right-hand sides are written with plain ``numpy`` so SciPy, diffrax (via ``jax.numpy`` substitution) and
torchdiffeq can all evaluate them; see :func:`rhs_for_backend`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    f: Callable  # f(t, y, xp) -> dy/dt, with xp the array module (numpy / jax.numpy / torch)
    y0: tuple[float, ...]
    t_span: tuple[float, float]
    stiff: bool
    jac: Callable | None = None  # jac(t, y) -> numpy matrix (SciPy only)
    exact: Callable | None = None  # exact(t) -> numpy vector, when known
    group: str = ""
    description: str = ""
    checkpoints: tuple[float, ...] = field(default_factory=tuple)  # extra times to compare at (besides t_end)
    scale: float = 1.0  # typical magnitude of y, for the mixed error norm

    @property
    def dim(self) -> int:
        return len(self.y0)


PROBLEMS: dict[str, Problem] = {}


def _add(p: Problem) -> None:
    PROBLEMS[p.name] = p


# ----------------------------------------------------------------------------- DETEST class A
_add(
    Problem(
        "A1",
        lambda t, y, xp: -y,
        (1.0,),
        (0.0, 20.0),
        False,
        exact=lambda t: np.array([np.exp(-t)]),
        group="DETEST-A",
        description="y' = -y",
    )
)
_add(
    Problem(
        "A2",
        lambda t, y, xp: -(y**3) / 2,
        (1.0,),
        (0.0, 20.0),
        False,
        exact=lambda t: np.array([1 / np.sqrt(1 + t)]),
        group="DETEST-A",
        description="y' = -y^3/2",
    )
)
_add(
    Problem(
        "A3",
        lambda t, y, xp: y * xp.cos(t),
        (1.0,),
        (0.0, 20.0),
        False,
        exact=lambda t: np.array([np.exp(np.sin(t))]),
        group="DETEST-A",
        description="y' = y cos t",
    )
)
_add(
    Problem(
        "A4",
        lambda t, y, xp: y / 4 * (1 - y / 20),
        (1.0,),
        (0.0, 20.0),
        False,
        exact=lambda t: np.array([20 / (1 + 19 * np.exp(-t / 4))]),
        group="DETEST-A",
        description="logistic y' = y/4 (1 - y/20)",
    )
)
_add(
    Problem(
        "A5",
        lambda t, y, xp: (y - t) / (y + t),
        (4.0,),
        (0.0, 20.0),
        False,
        group="DETEST-A",
        description="y' = (y - t)/(y + t), spiral",
    )
)

# ----------------------------------------------------------------------------- DETEST class B
_add(
    Problem(
        "B1",
        lambda t, y, xp: xp.stack([2 * (y[0] - y[0] * y[1]), -(y[1] - y[0] * y[1])]),
        (1.0, 3.0),
        (0.0, 20.0),
        False,
        group="DETEST-B",
        description="two conflicting populations (Lotka-Volterra)",
    )
)
_B2 = np.array([[-1.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -1.0]])
_add(
    Problem(
        "B2",
        lambda t, y, xp: xp.stack([-y[0] + y[1], y[0] - 2 * y[1] + y[2], y[1] - y[2]]),
        (2.0, 0.0, 1.0),
        (0.0, 20.0),
        False,
        jac=lambda t, y: _B2,
        group="DETEST-B",
        description="linear chemical reaction, 3x3",
    )
)
_add(
    Problem(
        "B3",
        lambda t, y, xp: xp.stack([-y[0], y[0] - y[1] ** 2, y[1] ** 2]),
        (1.0, 0.0, 0.0),
        (0.0, 20.0),
        False,
        group="DETEST-B",
        description="nonlinear chemical reaction",
    )
)
_add(
    Problem(
        "B4",
        lambda t, y, xp: xp.stack(
            [
                -y[1] - y[0] * y[2] / xp.sqrt(y[0] ** 2 + y[1] ** 2),
                y[0] - y[1] * y[2] / xp.sqrt(y[0] ** 2 + y[1] ** 2),
                y[0] / xp.sqrt(y[0] ** 2 + y[1] ** 2),
            ]
        ),
        (3.0, 0.0, 0.0),
        (0.0, 20.0),
        False,
        group="DETEST-B",
        description="integral surface of a torus",
    )
)
_add(
    Problem(
        "B5",
        lambda t, y, xp: xp.stack([y[1] * y[2], -y[0] * y[2], -0.51 * y[0] * y[1]]),
        (0.0, 1.0, 1.0),
        (0.0, 20.0),
        False,
        group="DETEST-B",
        description="Euler equations of a rigid body without external forces",
    )
)

# ----------------------------------------------------------------------------- DETEST class C (decay chains)
_add(
    Problem(
        "C1",
        lambda t, y, xp: xp.stack([-y[0]] + [y[i - 1] - y[i] for i in range(1, 9)] + [y[8]]),
        (1.0,) + (0.0,) * 9,
        (0.0, 20.0),
        False,
        group="DETEST-C",
        description="radioactive decay chain, 10 species, equal rates",
    )
)
_add(
    Problem(
        "C2",
        lambda t, y, xp: xp.stack([-y[0]] + [(i) * y[i - 1] - (i + 1) * y[i] for i in range(1, 9)] + [9 * y[8]]),
        (1.0,) + (0.0,) * 9,
        (0.0, 20.0),
        False,
        group="DETEST-C",
        description="radioactive decay chain, increasing rates",
    )
)


# ----------------------------------------------------------------------------- DETEST class D (orbits)
def _orbit(eps: float) -> Problem:
    def f(t, y, xp):
        r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
        return xp.stack([y[2], y[3], -y[0] / r3, -y[1] / r3])

    def exact(t):
        # Kepler: period 2π, x = cos E - e, y = sqrt(1-e²) sin E, E - e sin E = t
        E = t
        for _ in range(60):
            E = E - (E - eps * np.sin(E) - t) / (1 - eps * np.cos(E))
        x, yy = np.cos(E) - eps, np.sqrt(1 - eps**2) * np.sin(E)
        dEdt = 1 / (1 - eps * np.cos(E))
        return np.array([x, yy, -np.sin(E) * dEdt, np.sqrt(1 - eps**2) * np.cos(E) * dEdt])

    return Problem(
        f"D{int(round((eps - 0.1) / 0.2)) + 1}",
        f,
        (1 - eps, 0.0, 0.0, np.sqrt((1 + eps) / (1 - eps))),
        (0.0, 20.0),
        False,
        exact=exact,
        group="DETEST-D",
        description=f"two-body orbit, eccentricity {eps}",
    )


for _e in (0.1, 0.3, 0.5, 0.7, 0.9):
    _add(_orbit(_e))

# ----------------------------------------------------------------------------- DETEST class E
_add(
    Problem(
        "E1",
        lambda t, y, xp: xp.stack([y[1], -(y[1] / (t + 1) + (1 - 0.25 / (t + 1) ** 2) * y[0])]),
        (0.6713967071418030, 0.09540051444747446),
        (0.0, 20.0),
        False,
        group="DETEST-E",
        description="Bessel equation of order 1/2 (shifted)",
    )
)
_add(
    Problem(
        "E2",
        lambda t, y, xp: xp.stack([y[1], (1 - y[0] ** 2) * y[1] - y[0]]),
        (2.0, 0.0),
        (0.0, 20.0),
        False,
        group="DETEST-E",
        description="Van der Pol, mu = 1",
    )
)
_add(
    Problem(
        "E3",
        lambda t, y, xp: xp.stack([y[1], y[0] ** 3 / 6 - y[0] + 2 * xp.sin(2.78535 * t)]),
        (0.0, 0.0),
        (0.0, 20.0),
        False,
        group="DETEST-E",
        description="Duffing equation",
    )
)
_add(
    Problem(
        "E4",
        lambda t, y, xp: xp.stack([y[1], 0.032 - 0.4 * y[1] ** 2]),
        (30.0, 0.0),
        (0.0, 20.0),
        False,
        group="DETEST-E",
        description="falling body",
    )
)
_add(
    Problem(
        "E5",
        lambda t, y, xp: xp.stack([y[1], xp.sqrt(1 + y[1] ** 2) / (25 - t)]),
        (0.0, 0.0),
        (0.0, 20.0),
        False,
        group="DETEST-E",
        description="linear pursuit",
    )
)


# ----------------------------------------------------------------------------- Hairer & Wanner stiff set
def _rober(t, y, xp):
    return xp.stack(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


def _rober_jac(t, y):
    return np.array(
        [
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [0.0, 6e7 * y[1], 0.0],
        ]
    )


_add(
    Problem(
        "ROBER",
        _rober,
        (1.0, 0.0, 0.0),
        (0.0, 1e11),
        True,
        jac=_rober_jac,
        group="Hairer-stiff",
        description="Robertson chemical kinetics to t = 1e11",
        checkpoints=(40.0, 1e4, 1e7),
    )
)


def _vdp_stiff(t, y, xp):
    mu = 1000.0
    return xp.stack([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])


_add(
    Problem(
        "VDPOL",
        _vdp_stiff,
        (2.0, 0.0),
        (0.0, 3000.0),
        True,
        jac=lambda t, y: np.array([[0.0, 1.0], [-2000.0 * y[0] * y[1] - 1.0, 1000.0 * (1 - y[0] ** 2)]]),
        group="Hairer-stiff",
        description="Van der Pol, mu = 1000, to t = 3000 (about two periods)",
        checkpoints=(800.0, 1600.0),
    )
)


def _hires(t, y, xp):
    return xp.stack(
        [
            -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007,
            1.71 * y[0] - 8.75 * y[1],
            -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
            8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
            -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
            -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
            280.0 * y[5] * y[7] - 1.81 * y[6],
            -280.0 * y[5] * y[7] + 1.81 * y[6],
        ]
    )


_add(
    Problem(
        "HIRES",
        _hires,
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057),
        (0.0, 321.8122),
        True,
        group="Hairer-stiff",
        description="High irradiance response of photomorphogenesis (Schäfer), 8 species",
        checkpoints=(5.0, 50.0),
    )
)


def _orego(t, y, xp):
    return xp.stack(
        [
            77.27 * (y[1] + y[0] * (1 - 8.375e-6 * y[0] - y[1])),
            (y[2] - (1 + y[0]) * y[1]) / 77.27,
            0.161 * (y[0] - y[2]),
        ]
    )


_add(
    Problem(
        "OREGO",
        _orego,
        (1.0, 2.0, 3.0),
        (0.0, 360.0),
        True,
        group="Hairer-stiff",
        description="Oregonator (Belousov-Zhabotinsky), one period",
        checkpoints=(100.0, 250.0),
        scale=1e3,
    )
)


def _akzo(t, y, xp):
    k1, k2, k3, k4, K, klA, Ks, pCO2, H = 18.7, 0.58, 0.09, 0.42, 34.4, 3.3, 115.83, 0.9, 737.0
    r1 = k1 * y[0] ** 4 * xp.sqrt(xp.abs(y[1]) + 1e-300)
    r2 = k2 * y[2] * y[3]
    r3 = k2 / K * y[0] * y[4]
    r4 = k3 * y[0] * y[3] ** 2
    r5 = k4 * y[5] ** 2 * xp.sqrt(xp.abs(y[1]) + 1e-300)
    Fin = klA * (pCO2 / H - y[1])
    return xp.stack(
        [
            -2 * r1 + r2 - r3 - r4,
            -0.5 * r1 - r4 - 0.5 * r5 + Fin,
            r1 - r2 + r3,
            -r2 + r3 - 2 * r4,
            r2 - r3 + r5,
            Ks * y[0] * y[3] - y[5],  # algebraic in the DAE form; treated as very stiff ODE here with small eps
        ]
    )


_add(
    Problem(
        "AKZO",
        lambda t, y, xp: _akzo(t, y, xp) * xp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1e4]),
        (0.444, 0.00123, 0.0, 0.007, 0.0, 115.83 * 0.444 * 0.007),
        (0.0, 180.0),
        True,
        group="Hairer-stiff",
        description="Chemical Akzo Nobel (index-1 DAE as stiff ODE, eps = 1e-4)",
        checkpoints=(20.0, 100.0),
    )
)


def _e5(t, y, xp):
    A, B, C, M = 7.89e-10, 1.1e7, 1.13e3, 1.0e6
    return xp.stack(
        [
            -A * y[0] - B * y[0] * y[2],
            A * y[0] - M * C * y[1] * y[2],
            A * y[0] - B * y[0] * y[2] - M * C * y[1] * y[2] + C * y[3],
            B * y[0] * y[2] - C * y[3],
        ]
    )


_add(
    Problem(
        "E5S",
        _e5,
        (1.76e-3, 0.0, 0.0, 0.0),
        (0.0, 1e13),
        True,
        group="Hairer-stiff",
        description="E5 (Enright-Hull), very stiff, tiny components, to t = 1e13",
        checkpoints=(1e3, 1e7, 1e10),
        scale=1e-3,
    )
)


def rhs_for_backend(problem: Problem, xp) -> Callable:
    """Return ``f(t, y)`` with the array module bound (numpy, jax.numpy or torch)."""
    f = problem.f
    return lambda t, y: f(t, y, xp)


def list_problems(stiff: bool | None = None, group: str | None = None) -> list[Problem]:
    out = []
    for p in PROBLEMS.values():
        if stiff is not None and p.stiff != stiff:
            continue
        if group and p.group != group:
            continue
        out.append(p)
    return out
