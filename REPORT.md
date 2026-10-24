# ODE solver accuracy audit

odeaudit 0.1.0 · Python 3.13.7 · macOS-26.3.1-arm64-arm-64bit-Mach-O · scipy 1.17.1, diffrax not installed, jax 0.11.2, torchdiffeq not installed, torch 2.9.1

Requested tolerances rtol ∈ {1e-03, 1e-06, 1e-09}, atol = rtol * 1e-3 * problem.scale. Wall-clock budget per run 20 s.

Each cell: *achieved error ÷ requested tolerance* in the solver's own mixed norm at the worst checkpoint, and the number of correct significant digits at the final time. ✅ ≤ 10× · ⚠️ 10–1000× · ❌ > 1000× while reporting success (a silent failure) · ⏱ did not finish in the budget · 🛑 solver reported failure · 💥 raised. “≤ref” means the error is below what the reference itself can resolve.

## DETEST-A

### A1 — y' = -y
t ∈ [0, 20], dim 1, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 0.0163× (1.8 digits) | ✅ 0.23× (3.6 digits) | ✅ 0.645× (6.2 digits) |
| `scipy.DOP853` | ✅ 0.000238× (3.6 digits) | ✅ 0.000261× (6.6 digits) | ✅ 0.00618× (8.2 digits) |
| `scipy.LSODA` | ✅ 0.0147× (1.8 digits) | ✅ 0.0732× (4.1 digits) | ✅ 0.0147× (7.8 digits) |
| `scipy.RK23` | ✅ 1.11× (-0.0 digits) | ✅ 1.23× (2.9 digits) | ✅ 1.41× (5.9 digits) |
| `scipy.RK45` | ✅ 0.0385× (1.4 digits) | ✅ 0.118× (3.9 digits) | ✅ 0.167× (6.8 digits) |
| `scipy.Radau` | ✅ 0.00113× (2.9 digits) | ✅ 0.00295× (5.5 digits) | ✅ 0.00292× (8.5 digits) |
| `scipy.odeint` | ✅ 0.0317× (1.5 digits) | ✅ 0.0727× (4.1 digits) | ✅ 0.0406× (7.4 digits) |

### A2 — y' = -y^3/2
t ∈ [0, 20], dim 1, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 0.199× (3.7 digits) | ✅ 1.96× (5.7 digits) | ✅ 9.44× (8.0 digits) |
| `scipy.DOP853` | ✅ 2.76e-05× (7.6 digits) | ✅ 0.00104× (9.0 digits) | ✅ 0.0061× (11.2 digits) |
| `scipy.LSODA` | ✅ 0.783× (3.1 digits) | ✅ 1.87× (5.7 digits) | ✅ 1.38× (8.9 digits) |
| `scipy.RK23` | ✅ 0.933× (3.0 digits) | ✅ 1.23× (5.9 digits) | ✅ 1.27× (8.9 digits) |
| `scipy.RK45` | ✅ 1.46× (2.8 digits) | ✅ 1.04× (6.0 digits) | ✅ 0.447× (9.3 digits) |
| `scipy.Radau` | ✅ 0.0388× (4.4 digits) | ✅ 0.0108× (8.0 digits) | ✅ 0.00143× (11.8 digits) |
| `scipy.odeint` | ✅ 0.816× (3.1 digits) | ✅ 1.9× (5.7 digits) | ✅ 1.38× (8.9 digits) |

### A3 — y' = y cos t
t ∈ [0, 20], dim 1, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ⚠️ 11.1× (2.0 digits) | ⚠️ 12× (4.9 digits) | ⚠️ 37.9× (7.4 digits) |
| `scipy.DOP853` | ✅ 0.529× (3.3 digits) | ✅ 0.419× (6.4 digits) | ⚠️ 11.7× (7.9 digits) |
| `scipy.LSODA` | ✅ 3.14× (2.5 digits) | ✅ 6.43× (5.2 digits) | ✅ 0.571× (9.2 digits) |
| `scipy.RK23` | ✅ 9.92× (2.0 digits) | ⚠️ 15.5× (4.8 digits) | ⚠️ 22.7× (7.6 digits) |
| `scipy.RK45` | ✅ 1.55× (2.8 digits) | ✅ 2.08× (5.7 digits) | ✅ 1.95× (8.7 digits) |
| `scipy.Radau` | ✅ 0.00533× (5.3 digits) | ✅ 0.0129× (7.9 digits) | ✅ 0.00351× (11.5 digits) |
| `scipy.odeint` | ✅ 3× (2.5 digits) | ✅ 6.69× (5.2 digits) | ✅ 0.567× (9.2 digits) |

### A4 — logistic y' = y/4 (1 - y/20)
t ∈ [0, 20], dim 1, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 0.223× (3.7 digits) | ✅ 1.8× (5.7 digits) | ✅ 0.676× (9.2 digits) |
| `scipy.DOP853` | ✅ 0.019× (4.7 digits) | ✅ 0.00246× (8.6 digits) | ✅ 0.0076× (11.1 digits) |
| `scipy.LSODA` | ✅ 0.189× (3.7 digits) | ✅ 0.0792× (7.1 digits) | ✅ 0.74× (9.1 digits) |
| `scipy.RK23` | ✅ 0.352× (3.5 digits) | ✅ 1.57× (5.8 digits) | ✅ 1.77× (8.8 digits) |
| `scipy.RK45` | ✅ 0.339× (3.5 digits) | ✅ 0.119× (6.9 digits) | ✅ 0.0375× (10.4 digits) |
| `scipy.Radau` | ✅ 0.00615× (5.2 digits) | ✅ 0.000106× (10.0 digits) | ✅ 4.87e-05× (13.3 digits) |
| `scipy.odeint` | ✅ 0.145× (3.8 digits) | ✅ 0.335× (6.5 digits) | ✅ 0.769× (9.1 digits) |

### A5 — y' = (y - t)/(y + t), spiral
t ∈ [0, 20], dim 1, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ⚠️ 122× (0.9 digits) | ⚠️ 11.1× (5.0 digits) | ⚠️ 448× (6.3 digits) |
| `scipy.DOP853` | ✅ 0.122× (3.9 digits) | ✅ 0.936× (6.0 digits) | ✅ 3.52× (8.5 digits) |
| `scipy.LSODA` | ⚠️ 36.8× (1.4 digits) | ⚠️ 60.5× (4.2 digits) | ⚠️ 12.7× (7.9 digits) |
| `scipy.RK23` | ⚠️ 32.9× (1.5 digits) | ⚠️ 104× (4.0 digits) | ⚠️ 146× (6.8 digits) |
| `scipy.RK45` | ✅ 5.95× (2.2 digits) | ⚠️ 14.4× (4.8 digits) | ⚠️ 13.4× (7.9 digits) |
| `scipy.Radau` | ✅ 0.815× (3.1 digits) | ✅ 0.0934× (7.0 digits) | ✅ 0.00745× (11.1 digits) |
| `scipy.odeint` | ⚠️ 36.9× (1.4 digits) | ⚠️ 60.5× (4.2 digits) | ⚠️ 12.7× (7.9 digits) |

## DETEST-B

### B1 — two conflicting populations (Lotka-Volterra)
t ∈ [0, 20], dim 2, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ⚠️ 66.2× (1.2 digits) | ⚠️ 246× (3.6 digits) | ⚠️ 459× (6.3 digits) |
| `scipy.DOP853` | ✅ 2.51× (2.6 digits) | ⚠️ 12× (4.9 digits) | ✅ 7.25× (8.1 digits) |
| `scipy.LSODA` | ⚠️ 65.7× (1.2 digits) | ⚠️ 101× (4.0 digits) | ⚠️ 43.1× (7.4 digits) |
| `scipy.RK23` | ⚠️ 110× (1.0 digits) | ⚠️ 103× (4.0 digits) | ⚠️ 102× (7.0 digits) |
| `scipy.RK45` | ⚠️ 124× (0.9 digits) | ✅ 7.53× (5.1 digits) | ✅ 4.93× (8.3 digits) |
| `scipy.Radau` | ✅ 1.92× (2.7 digits) | ✅ 0.403× (6.4 digits) | ✅ 0.0777× (10.1 digits) |
| `scipy.odeint` | ⚠️ 65.8× (1.2 digits) | ⚠️ 100× (4.0 digits) | ⚠️ 43× (7.4 digits) |

### B2 — linear chemical reaction, 3x3
t ∈ [0, 20], dim 3, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 0.00146× (5.8 digits) | ✅ 0.0166× (7.8 digits) | ✅ 0.0898× (10.0 digits) |
| `scipy.DOP853` | ✅ 0.0406× (4.4 digits) | ✅ 0.00679× (8.2 digits) | ✅ 0.000319× (12.5 digits) |
| `scipy.LSODA` | ✅ 0.000336× (6.5 digits) | ✅ 0.0208× (7.7 digits) | ✅ 0.0938× (10.0 digits) |
| `scipy.RK23` | ✅ 0.189× (3.7 digits) | ✅ 0.332× (6.5 digits) | ✅ 0.695× (9.2 digits) |
| `scipy.RK45` | ✅ 0.182× (3.7 digits) | ✅ 0.582× (6.2 digits) | ✅ 0.0999× (10.0 digits) |
| `scipy.Radau` | ✅ 8.27e-05× (7.1 digits) | ✅ 0.00132× (8.9 digits) | ✅ 0.00645× (11.2 digits) |
| `scipy.odeint` | ✅ 0.000601× (6.2 digits) | ✅ 0.0438× (7.4 digits) | ✅ 0.0972× (10.0 digits) |

### B3 — nonlinear chemical reaction
t ∈ [0, 20], dim 3, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 0.504× (2.3 digits) | ✅ 1.09× (3.7 digits) | ✅ 5.66× (6.3 digits) |
| `scipy.DOP853` | ✅ 0.00101× (3.2 digits) | ✅ 0.00416× (5.4 digits) | ✅ 0.0443× (7.4 digits) |
| `scipy.LSODA` | ✅ 0.235× (2.1 digits) | ✅ 0.262× (5.0 digits) | ✅ 0.0861× (8.4 digits) |
| `scipy.RK23` | ✅ 2× (-0.3 digits) | ✅ 1.57× (4.5 digits) | ✅ 1.6× (7.6 digits) |
| `scipy.RK45` | ✅ 0.0461× (1.4 digits) | ✅ 0.0856× (4.1 digits) | ✅ 0.27× (6.6 digits) |
| `scipy.Radau` | ✅ 0.253× (3.4 digits) | ✅ 0.0315× (6.5 digits) | ✅ 0.00443× (10.1 digits) |
| `scipy.odeint` | ✅ 0.0884× (1.5 digits) | ✅ 0.274× (5.0 digits) | ✅ 0.13× (8.6 digits) |

### B4 — integral surface of a torus
t ∈ [0, 20], dim 3, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ⚠️ 43.9× (1.4 digits) | ✅ 8.02× (5.1 digits) | ⚠️ 77.3× (7.1 digits) |
| `scipy.DOP853` | ⚠️ 19× (1.7 digits) | ⚠️ 18.9× (4.7 digits) | ⚠️ 21.6× (7.7 digits) |
| `scipy.LSODA` | ✅ 8.35× (2.1 digits) | ⚠️ 26.9× (4.6 digits) | ⚠️ 20.2× (7.7 digits) |
| `scipy.RK23` | ⚠️ 32.5× (1.5 digits) | ✅ 3.79× (5.4 digits) | ✅ 6.13× (8.2 digits) |
| `scipy.RK45` | ⚠️ 45.4× (1.3 digits) | ✅ 4.88× (5.3 digits) | ✅ 2.03× (8.7 digits) |
| `scipy.Radau` | ✅ 1.67× (2.8 digits) | ✅ 0.108× (7.0 digits) | ✅ 0.0445× (10.4 digits) |
| `scipy.odeint` | ✅ 8.36× (2.1 digits) | ⚠️ 26.8× (4.6 digits) | ⚠️ 20.1× (7.7 digits) |

### B5 — Euler equations of a rigid body without external forces
t ∈ [0, 20], dim 3, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ⚠️ 107× (1.0 digits) | ⚠️ 254× (3.6 digits) | ⚠️ 237× (6.6 digits) |
| `scipy.DOP853` | ✅ 4.56× (2.3 digits) | ✅ 3.14× (5.5 digits) | ✅ 1.98× (8.7 digits) |
| `scipy.LSODA` | ⚠️ 24.9× (1.6 digits) | ⚠️ 76.8× (4.1 digits) | ✅ 6.92× (8.2 digits) |
| `scipy.RK23` | ⚠️ 11.7× (1.9 digits) | ⚠️ 14.3× (4.8 digits) | ⚠️ 14.3× (7.8 digits) |
| `scipy.RK45` | ⚠️ 89.8× (1.0 digits) | ⚠️ 34.8× (4.5 digits) | ⚠️ 23.1× (7.6 digits) |
| `scipy.Radau` | ✅ 7.03× (2.2 digits) | ✅ 1.29× (5.9 digits) | ✅ 0.0433× (10.4 digits) |
| `scipy.odeint` | ⚠️ 24.8× (1.6 digits) | ⚠️ 76.7× (4.1 digits) | ✅ 6.69× (8.2 digits) |

## DETEST-C

### C1 — radioactive decay chain, 10 species, equal rates
t ∈ [0, 20], dim 10, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 1.48× (0.9 digits) | ✅ 8.37× (3.5 digits) | ⚠️ 28.8× (6.0 digits) |
| `scipy.DOP853` | ✅ 0.0928× (2.4 digits) | ✅ 0.236× (4.9 digits) | ✅ 0.351× (7.7 digits) |
| `scipy.LSODA` | ✅ 0.721× (1.8 digits) | ✅ 0.619× (5.2 digits) | ✅ 2.47× (6.9 digits) |
| `scipy.RK23` | ⚠️ 18.1× (0.6 digits) | ⚠️ 10.6× (3.9 digits) | ⚠️ 10.2× (7.0 digits) |
| `scipy.RK45` | ✅ 0.938× (1.1 digits) | ✅ 0.922× (4.4 digits) | ✅ 0.91× (7.5 digits) |
| `scipy.Radau` | ✅ 0.0337× (2.9 digits) | ✅ 0.00899× (6.8 digits) | ✅ 0.0016× (10.5 digits) |
| `scipy.odeint` | ✅ 0.725× (1.8 digits) | ✅ 0.809× (4.8 digits) | ✅ 2.51× (6.9 digits) |

### C2 — radioactive decay chain, increasing rates
t ∈ [0, 20], dim 10, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 0.0352× (1.5 digits) | ✅ 0.103× (4.0 digits) | ✅ 0.609× (6.2 digits) |
| `scipy.DOP853` | ✅ 0.11× (1.0 digits) | ✅ 2.34× (2.6 digits) | ✅ 0.0107× (8.0 digits) |
| `scipy.LSODA` | ✅ 0.0138× (1.9 digits) | ✅ 0.17× (3.8 digits) | ✅ 0.13× (6.9 digits) |
| `scipy.RK23` | ✅ 4.34× (-0.6 digits) | ✅ 1.43× (2.8 digits) | ✅ 1.52× (5.8 digits) |
| `scipy.RK45` | ✅ 0.897× (0.0 digits) | ✅ 0.681× (3.2 digits) | ✅ 0.153× (6.8 digits) |
| `scipy.Radau` | ✅ 0.00187× (2.7 digits) | ✅ 0.013× (5.6 digits) | ✅ 0.00215× (8.7 digits) |
| `scipy.odeint` | ✅ 0.016× (1.8 digits) | ✅ 0.174× (3.8 digits) | ✅ 0.28× (6.6 digits) |

## DETEST-D

### D1 — two-body orbit, eccentricity 0.1
t ∈ [0, 20], dim 4, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ❌ 1.39e+03× (-0.1 digits) | ❌ 3.87e+03× (2.4 digits) | ❌ 9.08e+03× (5.0 digits) |
| `scipy.DOP853` | ⚠️ 18.4× (1.7 digits) | ⚠️ 86.2× (4.1 digits) | ⚠️ 19× (7.7 digits) |
| `scipy.LSODA` | ⚠️ 209× (0.7 digits) | ❌ 1.54e+03× (2.8 digits) | ⚠️ 356× (6.4 digits) |
| `scipy.RK23` | ⚠️ 430× (0.4 digits) | ⚠️ 626× (3.2 digits) | ⚠️ 616× (6.2 digits) |
| `scipy.RK45` | ❌ 4.89e+03× (-0.7 digits) | ⚠️ 582× (3.2 digits) | ⚠️ 323× (6.5 digits) |
| `scipy.Radau` | ⚠️ 27.2× (1.6 digits) | ✅ 4.55× (5.3 digits) | ✅ 0.862× (9.1 digits) |
| `scipy.odeint` | ⚠️ 209× (0.7 digits) | ❌ 1.54e+03× (2.8 digits) | ⚠️ 356× (6.4 digits) |

### D2 — two-body orbit, eccentricity 0.3
t ∈ [0, 20], dim 4, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ❌ 1.76e+03× (-0.2 digits) | ❌ 7.2e+03× (2.1 digits) | ❌ 1.92e+04× (4.7 digits) |
| `scipy.DOP853` | ⚠️ 35.9× (1.4 digits) | ⚠️ 640× (3.2 digits) | ⚠️ 444× (6.3 digits) |
| `scipy.LSODA` | ❌ 3.34e+03× (-0.5 digits) | ❌ 3.8e+03× (2.4 digits) | ⚠️ 814× (6.1 digits) |
| `scipy.RK23` | ❌ 1.18e+03× (-0.1 digits) | ❌ 1.25e+03× (2.9 digits) | ❌ 1.22e+03× (5.9 digits) |
| `scipy.RK45` | ❌ 4.14e+03× (-0.6 digits) | ❌ 1.91e+03× (2.7 digits) | ⚠️ 615× (6.2 digits) |
| `scipy.Radau` | ⚠️ 53.8× (1.3 digits) | ✅ 7.94× (5.1 digits) | ✅ 1.25× (8.9 digits) |
| `scipy.odeint` | ❌ 3.34e+03× (-0.5 digits) | ❌ 3.8e+03× (2.4 digits) | ⚠️ 814× (6.1 digits) |

### D3 — two-body orbit, eccentricity 0.5
t ∈ [0, 20], dim 4, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ❌ 3.62e+03× (-0.6 digits) | ❌ 1.11e+04× (1.9 digits) | ❌ 1.55e+04× (4.8 digits) |
| `scipy.DOP853` | ⚠️ 33.5× (1.5 digits) | ❌ 1.69e+03× (2.8 digits) | ⚠️ 838× (6.1 digits) |
| `scipy.LSODA` | ❌ 4.93e+03× (-0.7 digits) | ❌ 9.54e+03× (2.0 digits) | ❌ 7.04e+03× (5.1 digits) |
| `scipy.RK23` | ⚠️ 807× (0.1 digits) | ⚠️ 732× (3.1 digits) | ⚠️ 694× (6.2 digits) |
| `scipy.RK45` | ❌ 2.26e+03× (-0.4 digits) | ⚠️ 179× (3.7 digits) | ❌ 1.11e+03× (5.9 digits) |
| `scipy.Radau` | ⚠️ 75.1× (1.1 digits) | ✅ 9.98× (5.0 digits) | ✅ 1.88× (8.7 digits) |
| `scipy.odeint` | ❌ 4.93e+03× (-0.7 digits) | ❌ 9.54e+03× (2.0 digits) | ❌ 7.04e+03× (5.1 digits) |

### D4 — two-body orbit, eccentricity 0.7
t ∈ [0, 20], dim 4, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ❌ 1.58e+03× (-0.2 digits) | ❌ 1.01e+03× (3.0 digits) | ❌ 4.24e+03× (5.4 digits) |
| `scipy.DOP853` | ⚠️ 70.7× (1.1 digits) | ⚠️ 299× (3.5 digits) | ⚠️ 269× (6.6 digits) |
| `scipy.LSODA` | ❌ 1.56e+03× (-0.2 digits) | ⚠️ 76.7× (4.1 digits) | ⚠️ 910× (6.0 digits) |
| `scipy.RK23` | ⚠️ 11.1× (2.0 digits) | ⚠️ 117× (3.9 digits) | ⚠️ 123× (6.9 digits) |
| `scipy.RK45` | ⚠️ 507× (0.3 digits) | ⚠️ 256× (3.6 digits) | ⚠️ 357× (6.4 digits) |
| `scipy.Radau` | ⚠️ 28.3× (1.5 digits) | ✅ 4.42× (5.4 digits) | ✅ 0.761× (9.1 digits) |
| `scipy.odeint` | ❌ 1.56e+03× (-0.2 digits) | ⚠️ 76.5× (4.1 digits) | ⚠️ 910× (6.0 digits) |

### D5 — two-body orbit, eccentricity 0.9
t ∈ [0, 20], dim 4, reference: analytic (uncertainty 1e-15)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ❌ 3.46e+03× (-0.5 digits) | ❌ 4.88e+03× (2.3 digits) | ⚠️ 784× (6.1 digits) |
| `scipy.DOP853` | ⚠️ 114× (0.9 digits) | ⚠️ 334× (3.5 digits) | ⚠️ 318× (6.5 digits) |
| `scipy.LSODA` | ❌ 1.61e+03× (-0.2 digits) | ❌ 1.7e+03× (2.8 digits) | ❌ 4.56e+03× (5.3 digits) |
| `scipy.RK23` | ⚠️ 374× (0.4 digits) | ⚠️ 682× (3.2 digits) | ⚠️ 686× (6.2 digits) |
| `scipy.RK45` | ⚠️ 570× (0.2 digits) | ⚠️ 479× (3.3 digits) | ⚠️ 462× (6.3 digits) |
| `scipy.Radau` | ⚠️ 35.9× (1.4 digits) | ✅ 6.82× (5.2 digits) | ✅ 1.13× (8.9 digits) |
| `scipy.odeint` | ❌ 1.61e+03× (-0.2 digits) | ❌ 1.7e+03× (2.8 digits) | ❌ 4.56e+03× (5.3 digits) |

## DETEST-E

### E1 — Bessel equation of order 1/2 (shifted)
t ∈ [0, 20], dim 2, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 3.88× (2.4 digits) | ⚠️ 47.1× (4.3 digits) | ⚠️ 148× (6.8 digits) |
| `scipy.DOP853` | ✅ 2.65× (2.6 digits) | ✅ 2.44× (5.6 digits) | ✅ 1.74× (8.8 digits) |
| `scipy.LSODA` | ⚠️ 10.1× (2.0 digits) | ⚠️ 24.7× (4.6 digits) | ✅ 8.12× (8.1 digits) |
| `scipy.RK23` | ⚠️ 20.5× (1.7 digits) | ⚠️ 23.7× (4.6 digits) | ⚠️ 23.3× (7.6 digits) |
| `scipy.RK45` | ✅ 5.61× (2.2 digits) | ✅ 3.16× (5.5 digits) | ✅ 3.74× (8.4 digits) |
| `scipy.Radau` | ✅ 0.439× (3.4 digits) | ✅ 0.0486× (7.3 digits) | ✅ 0.00905× (11.0 digits) |
| `scipy.odeint` | ⚠️ 10.1× (2.0 digits) | ⚠️ 24.8× (4.6 digits) | ✅ 8.21× (8.1 digits) |

### E2 — Van der Pol, mu = 1
t ∈ [0, 20], dim 2, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ⚠️ 336× (0.5 digits) | ⚠️ 705× (3.1 digits) | ❌ 1.53e+03× (5.8 digits) |
| `scipy.DOP853` | ✅ 4.87× (2.3 digits) | ⚠️ 201× (3.7 digits) | ✅ 3.64× (8.4 digits) |
| `scipy.LSODA` | ⚠️ 16.9× (1.8 digits) | ⚠️ 248× (3.6 digits) | ⚠️ 859× (6.1 digits) |
| `scipy.RK23` | ⚠️ 118× (0.9 digits) | ⚠️ 54.8× (4.3 digits) | ⚠️ 36.5× (7.4 digits) |
| `scipy.RK45` | ❌ 1.65e+03× (-0.2 digits) | ⚠️ 194× (3.7 digits) | ⚠️ 103× (7.0 digits) |
| `scipy.Radau` | ⚠️ 13.4× (1.9 digits) | ✅ 0.243× (6.6 digits) | ✅ 0.0412× (10.4 digits) |
| `scipy.odeint` | ⚠️ 16.6× (1.8 digits) | ⚠️ 248× (3.6 digits) | ⚠️ 859× (6.1 digits) |

### E3 — Duffing equation
t ∈ [0, 20], dim 2, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ⚠️ 285× (0.5 digits) | ⚠️ 989× (3.0 digits) | ❌ 3.15e+03× (5.5 digits) |
| `scipy.DOP853` | ⚠️ 32.3× (1.5 digits) | ✅ 4.85× (5.3 digits) | ✅ 8.07× (8.1 digits) |
| `scipy.LSODA` | ⚠️ 15.7× (1.8 digits) | ⚠️ 210× (3.7 digits) | ⚠️ 82.7× (7.1 digits) |
| `scipy.RK23` | ⚠️ 111× (0.9 digits) | ⚠️ 122× (3.9 digits) | ⚠️ 122× (6.9 digits) |
| `scipy.RK45` | ⚠️ 769× (0.1 digits) | ⚠️ 187× (3.7 digits) | ⚠️ 65.1× (7.2 digits) |
| `scipy.Radau` | ✅ 0.125× (3.9 digits) | ✅ 0.0151× (7.8 digits) | ✅ 0.000972× (12.0 digits) |
| `scipy.odeint` | ⚠️ 15.9× (1.8 digits) | ⚠️ 210× (3.7 digits) | ⚠️ 82.7× (7.1 digits) |

### E4 — falling body
t ∈ [0, 20], dim 2, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 0.164× (3.8 digits) | ✅ 2.5× (5.6 digits) | ✅ 2.57× (8.6 digits) |
| `scipy.DOP853` | ✅ 0.00456× (5.3 digits) | ✅ 0.00214× (8.7 digits) | ✅ 0.0249× (10.6 digits) |
| `scipy.LSODA` | ✅ 0.0277× (4.6 digits) | ✅ 0.0879× (7.1 digits) | ✅ 2.42× (8.6 digits) |
| `scipy.RK23` | ✅ 4.21× (2.4 digits) | ✅ 2.18× (5.7 digits) | ✅ 2.16× (8.7 digits) |
| `scipy.RK45` | ✅ 0.0526× (4.3 digits) | ✅ 0.142× (6.8 digits) | ✅ 0.239× (9.6 digits) |
| `scipy.Radau` | ✅ 0.0751× (4.1 digits) | ✅ 0.00176× (8.8 digits) | ✅ 8.06e-05× (13.1 digits) |
| `scipy.odeint` | ✅ 0.0124× (4.9 digits) | ✅ 0.0917× (7.0 digits) | ✅ 3.21× (8.5 digits) |

### E5 — linear pursuit
t ∈ [0, 20], dim 2, reference: taylor (uncertainty 1e-19)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.BDF` | ✅ 4.17× (2.4 digits) | ⚠️ 14.8× (4.8 digits) | ⚠️ 33.2× (7.5 digits) |
| `scipy.DOP853` | ✅ 0.106× (4.0 digits) | ✅ 0.0767× (7.1 digits) | ✅ 0.0885× (10.1 digits) |
| `scipy.LSODA` | ✅ 4.02× (2.4 digits) | ✅ 9.47× (5.0 digits) | ⚠️ 16.4× (7.8 digits) |
| `scipy.RK23` | ✅ 0.646× (3.2 digits) | ✅ 0.713× (6.1 digits) | ✅ 0.696× (9.2 digits) |
| `scipy.RK45` | ✅ 0.151× (3.8 digits) | ✅ 0.0967× (7.0 digits) | ✅ 0.15× (9.8 digits) |
| `scipy.Radau` | ✅ 0.062× (4.2 digits) | ✅ 0.0107× (8.0 digits) | ✅ 0.0024× (11.6 digits) |
| `scipy.odeint` | ✅ 3.81× (2.4 digits) | ✅ 9.47× (5.0 digits) | ⚠️ 16.4× (7.8 digits) |

## Hairer-stiff

### ROBER — Robertson chemical kinetics to t = 1e11
t ∈ [0, 1e+11], dim 3, reference: consensus (uncertainty 4e-08)

| solver | rtol 1e-03 | rtol 1e-06 | rtol 1e-09 |
|---|---|---|---|
| `scipy.RK45` | 💥 crashed | ⏱ timeout | ⏱ timeout |

## Silent failures

Solver reported success; the answer is more than 1000× worse than requested.

| problem | solver | rtol | error/rtol | correct digits |
|---|---|---|---|---|
| D2 | `scipy.BDF` | 1e-09 | 1.92e+04 | 4.7 |
| D3 | `scipy.BDF` | 1e-09 | 1.55e+04 | 4.8 |
| D3 | `scipy.BDF` | 1e-06 | 1.11e+04 | 1.9 |
| D3 | `scipy.odeint` | 1e-06 | 9.54e+03 | 2.0 |
| D3 | `scipy.LSODA` | 1e-06 | 9.54e+03 | 2.0 |
| D1 | `scipy.BDF` | 1e-09 | 9.08e+03 | 5.0 |
| D2 | `scipy.BDF` | 1e-06 | 7.2e+03 | 2.1 |
| D3 | `scipy.LSODA` | 1e-09 | 7.04e+03 | 5.1 |
| D3 | `scipy.odeint` | 1e-09 | 7.04e+03 | 5.1 |
| D3 | `scipy.LSODA` | 1e-03 | 4.93e+03 | -0.7 |
| D3 | `scipy.odeint` | 1e-03 | 4.93e+03 | -0.7 |
| D1 | `scipy.RK45` | 1e-03 | 4.89e+03 | -0.7 |
| D5 | `scipy.BDF` | 1e-06 | 4.88e+03 | 2.3 |
| D5 | `scipy.odeint` | 1e-09 | 4.56e+03 | 5.3 |
| D5 | `scipy.LSODA` | 1e-09 | 4.56e+03 | 5.3 |
| D4 | `scipy.BDF` | 1e-09 | 4.24e+03 | 5.4 |
| D2 | `scipy.RK45` | 1e-03 | 4.14e+03 | -0.6 |
| D1 | `scipy.BDF` | 1e-06 | 3.87e+03 | 2.4 |
| D2 | `scipy.LSODA` | 1e-06 | 3.8e+03 | 2.4 |
| D2 | `scipy.odeint` | 1e-06 | 3.8e+03 | 2.4 |
| D3 | `scipy.BDF` | 1e-03 | 3.62e+03 | -0.6 |
| D5 | `scipy.BDF` | 1e-03 | 3.46e+03 | -0.5 |
| D2 | `scipy.odeint` | 1e-03 | 3.34e+03 | -0.5 |
| D2 | `scipy.LSODA` | 1e-03 | 3.34e+03 | -0.5 |
| E3 | `scipy.BDF` | 1e-09 | 3.15e+03 | 5.5 |
| D3 | `scipy.RK45` | 1e-03 | 2.26e+03 | -0.4 |
| D2 | `scipy.RK45` | 1e-06 | 1.91e+03 | 2.7 |
| D2 | `scipy.BDF` | 1e-03 | 1.76e+03 | -0.2 |
| D5 | `scipy.odeint` | 1e-06 | 1.7e+03 | 2.8 |
| D5 | `scipy.LSODA` | 1e-06 | 1.7e+03 | 2.8 |
| D3 | `scipy.DOP853` | 1e-06 | 1.69e+03 | 2.8 |
| E2 | `scipy.RK45` | 1e-03 | 1.65e+03 | -0.2 |
| D5 | `scipy.LSODA` | 1e-03 | 1.61e+03 | -0.2 |
| D5 | `scipy.odeint` | 1e-03 | 1.61e+03 | -0.2 |
| D4 | `scipy.BDF` | 1e-03 | 1.58e+03 | -0.2 |
| D4 | `scipy.odeint` | 1e-03 | 1.56e+03 | -0.2 |
| D4 | `scipy.LSODA` | 1e-03 | 1.56e+03 | -0.2 |
| D1 | `scipy.odeint` | 1e-06 | 1.54e+03 | 2.8 |
| D1 | `scipy.LSODA` | 1e-06 | 1.54e+03 | 2.8 |
| E2 | `scipy.BDF` | 1e-09 | 1.53e+03 | 5.8 |
| D1 | `scipy.BDF` | 1e-03 | 1.39e+03 | -0.1 |
| D2 | `scipy.RK23` | 1e-06 | 1.25e+03 | 2.9 |
| D2 | `scipy.RK23` | 1e-09 | 1.22e+03 | 5.9 |
| D2 | `scipy.RK23` | 1e-03 | 1.18e+03 | -0.1 |
| D3 | `scipy.RK45` | 1e-09 | 1.11e+03 | 5.9 |
| D4 | `scipy.BDF` | 1e-06 | 1.01e+03 | 3.0 |

## Did not finish within the budget

| problem | solver | rtol |
|---|---|---|
| ROBER | `scipy.RK45` | 1e-06 |
| ROBER | `scipy.RK45` | 1e-09 |

## Reported failures / crashes

| problem | solver | rtol | status | message |
|---|---|---|---|---|
| ROBER | `scipy.RK45` | 1e-03 | crashed | AttributeError: 'list' object has no attribute 'size' |
