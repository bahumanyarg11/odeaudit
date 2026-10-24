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
### Headline findings (SciPy 1.17.1 `solve_ivp` + `odeint`, this machine; diffrax/torchdiffeq rows pending)

- **Explicit methods never finish the stiff problems.** RK45, RK23 and DOP853 hit the 20 s budget on ROBER, VDPOL, OREGO and
  AKZO at every tolerance (RK45 on ROBER at rtol 1e-3 fails outright: "Required step size is less than spacing between numbers").
  `solve_ivp`'s default method is RK45; nothing warns the user that the problem is stiff.
- **BDF on Van der Pol (μ = 1000): a silent failure and a hard failure.** At rtol 1e-3 BDF reports success with an answer
  2200× outside tolerance (1.5 correct digits); at rtol 1e-9 it stops with a step-size underflow. LSODA and `odeint` finish
  but are 350–770× outside tolerance at every rtol.
- **HIRES and OREGO: implicit methods deliver 10–200× the requested error** (BDF worst: OREGO at rtol 1e-9, 204×), while
  Radau stays within 10× everywhere; DOP853 on HIRES is the only explicit run that completes the stiff set.
- **Non-stiff DETEST (462 runs, no failures):** Radau is essentially exact, DOP853 and RK45 stay near their tolerance
  (median 2.4× and 4× at rtol 1e-6), RK23, BDF, LSODA and `odeint` run 13–26× loose. On the two-body orbits D1–D5 every
  solver ends 100–10000× outside tolerance after three revolutions: expected global error growth, see Caveats.
- E5 (Enright–Hull) is excluded until a trustworthy reference exists: Radau at rtol 1e-12 cannot complete it.
<!-- /HEADLINES -->

## Caveats

- Stiff references are a tight Radau solve, not an exact solution; errors below the reported
  reference uncertainty are shown as "≤ref" and never counted against a solver.
- `atol = rtol × 1e-3 × scale` for every solver, so the comparison is between tolerance *interpretations*,
  not atol choices. E5 and OREGO carry a problem `scale` because their components span many orders.
- Chemical Akzo Nobel is an index-1 DAE; it is run here as a stiff ODE with the algebraic equation relaxed (ε = 1e-4).
- Timings are single-process wall clock and include JIT compilation for diffrax.
- Local error control does not bound *global* error. On the orbit problems D1–D5 (about three revolutions)
  phase error accumulates and every solver ends up 100–10000× outside its tolerance; that is a property of
  the problem class, shown here because it is what a user gets, not a defect of any one library. Compare
  solvers against each other on those rows rather than against the ≤ 10× line.

## Adding a problem

A `Problem` is a right-hand side written against an array module (`xp` = numpy, jax.numpy or torch),
`y0`, `t_span`, optional Jacobian and closed form, and checkpoint times. See `odeaudit/problems.py`.

## License

BSD 3-Clause.
