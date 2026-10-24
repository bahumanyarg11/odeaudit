import numpy as np
import pytest

from odeaudit.metrics import assess
from odeaudit.problems import PROBLEMS, rhs_for_backend
from odeaudit.references import compute_reference
from odeaudit.solvers import solver_by_name


def test_all_rhs_evaluate_at_y0():
    for p in PROBLEMS.values():
        f = rhs_for_backend(p, np)
        dy = np.asarray(f(p.t_span[0], np.asarray(p.y0, float)), float)
        assert dy.shape == (p.dim,), p.name
        assert np.all(np.isfinite(dy)), p.name


def test_analytic_references_satisfy_the_ode():
    # d/dt exact(t) ≈ f(t, exact(t)) by finite differences, for every problem with a closed form
    for p in PROBLEMS.values():
        if p.exact is None:
            continue
        f = rhs_for_backend(p, np)
        t, h = 3.0, 1e-6
        lhs = (np.asarray(p.exact(t + h)) - np.asarray(p.exact(t - h))) / (2 * h)
        rhs = np.asarray(f(t, np.asarray(p.exact(t))), float)
        assert np.allclose(lhs, rhs, rtol=1e-6, atol=1e-7), p.name
        assert np.allclose(np.asarray(p.exact(p.t_span[0])), p.y0, atol=1e-12), p.name


def test_taylor_reference_matches_analytic_on_A1():
    p = PROBLEMS["A1"]
    from odeaudit.references import _taylor

    vals, unc = _taylor(p, [20.0])
    assert vals[0][0] == pytest.approx(np.exp(-20.0), rel=1e-14)


def test_consensus_reference_rober():
    ref = compute_reference(PROBLEMS["ROBER"])
    assert ref.source == "consensus"
    y_end = ref.values[-1]
    assert y_end[0] == pytest.approx(0.0, abs=1e-6) and y_end[2] == pytest.approx(1.0, abs=1e-6)
    assert ref.uncertainty < 1e-6


def test_assess_verdicts():
    ref = np.array([[1.0, 2.0]])
    good = assess(np.array([[1.0 + 1e-9, 2.0]]), ref, 1e-6, 1e-9, 1.0, 1e-15)
    assert good["verdict"] == "accurate"
    bad = assess(np.array([[1.1, 2.0]]), ref, 1e-6, 1e-9, 1.0, 1e-15)
    assert bad["verdict"] == "silent_failure" and bad["tol_ratio"] > 1000


def test_scipy_adapter_rk45_on_A1():
    p = PROBLEMS["A1"]
    run = solver_by_name("scipy.RK45")(p, [20.0], 1e-9, 1e-12)
    assert run.success and run.values[0][0] == pytest.approx(np.exp(-20.0), rel=1e-6)


def test_small_audit_runs(tmp_path):
    from odeaudit.report import build_report
    from odeaudit.run import run_audit

    payload = run_audit(["A1", "B1"], ["scipy"], (1e-6,), timeout=30, out_dir=tmp_path, verbose=False)
    assert len(payload["cases"]) == 12  # 2 problems x 6 scipy methods
    text = build_report(tmp_path / "results.json", tmp_path / "REPORT.md")
    assert "A1" in text and "scipy.RK45" in text
