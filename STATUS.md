# odeaudit build status

Resume: read this, `git log --oneline | head`, `PYTHONPATH=. .venv/bin/python -m pytest -q`, continue with the first unchecked item.
Lives on the Mac internal disk (`09_PROJECTS/odeaudit`, source + small JSON only) because the Seagate drive was unplugged when it was
started; `.venv` reuses the system numpy/scipy/mpmath (`--system-site-packages`). diffrax/torchdiffeq need the JAX/PyTorch venv on the
Seagate (`specaudit/.venv`: `uv pip install diffrax torchdiffeq` into it, then run with that interpreter). No A6000. Commit as the user only.

## Done
- [x] Problems: DETEST A1–A5, B1–B5, C1–C2, D1–D5, E1–E5; Hairer ROBER, VDPOL, HIRES, OREGO, AKZO (DAE as stiff ODE), E5
- [x] References: analytic / mpmath Taylor (25 digits) / Radau–BDF consensus with reported uncertainty, cached
- [x] Adapters: scipy methods, odeint, diffrax, torchdiffeq; per-case child process with wall-clock budget
- [x] Mixed-norm error ÷ rtol metric, verdicts, incremental results.json with resume, Markdown report, CLI, 7 tests

## Queue
1. [x] Stiff-set run (scipy + odeint, 20 s budget) → REPORT.md; headlines in README
2. [ ] diffrax + torchdiffeq runs once the Seagate drive is back
3. [x] README headline findings; published at github.com/bahumanyarg11/odeaudit. Upstream (SciPy) only after hand verification of BDF/VDPOL and a check against SciPy's own benchmarks
4. [ ] Later: Pollution (20-dim) and Ring Modulator from the Bari test set; float32 runs for torchdiffeq/diffrax

## Log
- 2026-10-24 built; DETEST run with SciPy/odeint completed (see results/).
- E5 excluded: Radau rtol 1e-12 cannot complete it (step underflow); kept as E5_PROBLEM for anyone with a reference. 567 SciPy/odeint cases in results/.
