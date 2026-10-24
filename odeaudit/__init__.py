"""odeaudit: achieved-vs-requested accuracy of ODE solvers on the Hairer stiff and DETEST test sets."""

from ._version import __version__
from .problems import PROBLEMS, Problem
from .run import run_audit

__all__ = ["PROBLEMS", "Problem", "run_audit", "__version__"]
