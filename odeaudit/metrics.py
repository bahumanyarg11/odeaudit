"""How a run is judged.

For each checkpoint the error is measured in the solver's own mixed norm,
``e = max_j |y_j - ref_j| / (atol + rtol |ref_j|)``: the quantity a tolerance-respecting solver keeps
near 1 per step, so a global value of a few is the normal price of error accumulation, and a value of
1e3 means the answer is a thousand times worse than the user asked for. We also report ``scd``, the
number of significant correct digits at the final time (the Bari test-set convention), with the
problem's ``scale`` as the floor for relative error so tiny components do not dominate.

Verdicts (on the worst checkpoint):

* ``accurate``        tol_ratio ≤ 10
* ``loose``           10 < tol_ratio ≤ 1000
* ``silent_failure``  tol_ratio > 1000 while the solver reported success
* ``at_reference``    the error is below the reference's own uncertainty (cannot be resolved)
"""

from __future__ import annotations

import math

import numpy as np


def assess(values: np.ndarray, ref: np.ndarray, rtol: float, atol: float, scale: float, ref_unc: float) -> dict:
    if values.shape != ref.shape:
        return {"verdict": "shape_mismatch", "scd": math.nan, "tol_ratio": math.inf}
    err = np.abs(values - ref)
    mixed = err / (atol + rtol * np.abs(ref))
    rel = err / np.maximum(np.abs(ref), 1e-6 * scale)
    tol_ratio = float(np.max(mixed))
    rel_final = float(np.max(rel[-1]))
    rel_max = float(np.max(rel))
    scd = -math.log10(rel_final) if rel_final > 0 else 16.0
    if rel_max <= ref_unc * 10:
        verdict = "at_reference"
    elif tol_ratio <= 10:
        verdict = "accurate"
    elif tol_ratio <= 1000:
        verdict = "loose"
    else:
        verdict = "silent_failure"
    worst = int(np.argmax(mixed.max(axis=1)))
    return {
        "verdict": verdict,
        "tol_ratio": tol_ratio,
        "rel_err_final": rel_final,
        "rel_err_max": rel_max,
        "scd": float(min(scd, 16.0)),
        "worst_checkpoint": worst,
        "ref_uncertainty": ref_unc,
    }
