# odeaudit

**What accuracy do you actually get from an ODE solver at the tolerance you asked for?**

odeaudit runs SciPy's `solve_ivp` methods (RK45, RK23, DOP853, Radau, BDF, LSODA), `odeint`,
diffrax (Tsit5, Dopri5, Dopri8, Kvaerno5) and torchdiffeq (dopri5, dopri8, bosh3, adaptive_heun) on the
classic test sets — the **DETEST** non-stiff problems (Hull–Enright–Fellen–Sedgwick classes A–E) and the
**Hairer–Wanner stiff set** (ROBER, VDPOL μ=1000, HIRES, OREGO, Chemical Akzo Nobel, E5) — at
rtol = 1e-3, 1e-6, 1e-9, and compares every result with a reference it can vouch for:

| reference | used for | accuracy |
|---|---|---|
| closed form (Kepler's equation for the orbits) | DETEST A1–A4, D1–D5 | ~1e-15 |
| mpmath Taylor-series integration at 25 digits | the other non-stiff problems | ~1e-19 |
| Radau at rtol 1e-12, cross-checked against BDF | stiff problems | reported per problem (1e-9…1e-11) |

The headline number per cell is **achieved error ÷ requested tolerance** in the solver's own mixed
norm, max over checkpoints: ≤ 10 is what error accumulation normally costs, 10–1000 is loose, and
> 1000 while the solver reports success is a *silent failure*. Explicit solvers on stiff problems get a
wall-clock budget; not finishing is recorded (⏱), because that is the behaviour a user hits.

```bash
pip install odeaudit                  # numpy, scipy, mpmath
pip install "odeaudit[jax,torch]"     # + diffrax, torchdiffeq
odeaudit list
odeaudit run -l scipy,odeint --timeout 60 -o results       # resumable, saves after every case
odeaudit report results/results.json -o REPORT.md
```

```python
from odeaudit import PROBLEMS, run_audit
run_audit(["ROBER", "HIRES"], ["scipy"], rtols=(1e-6,), timeout=30, out_dir="out")
```

## Results

See [REPORT.md](REPORT.md) for every problem × solver × tolerance on this machine.

<!-- HEADLINES -->

## Caveats

- Stiff references are a tight Radau solve, not an exact solution; errors below the reported
  reference uncertainty are shown as "≤ref" and never counted against a solver.
- `atol = rtol × 1e-3 × scale` for every solver, so the comparison is between tolerance *interpretations*,
  not atol choices. E5 and OREGO carry a problem `scale` because their components span many orders.
- Chemical Akzo Nobel is an index-1 DAE; it is run here as a stiff ODE with the algebraic equation relaxed (ε = 1e-4).
- Timings are single-process wall clock and include JIT compilation for diffrax.

## Adding a problem

A `Problem` is a right-hand side written against an array module (`xp` = numpy, jax.numpy or torch),
`y0`, `t_span`, optional Jacobian and closed form, and checkpoint times. See `odeaudit/problems.py`.

## License

BSD 3-Clause.
