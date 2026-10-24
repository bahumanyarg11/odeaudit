"""Markdown report from results.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .problems import PROBLEMS

GLYPH = {
    "accurate": "✅",
    "at_reference": "✅",
    "loose": "⚠️",
    "silent_failure": "❌",
    "timeout": "⏱",
    "failed": "🛑",
    "crashed": "💥",
}


def _cell(c: dict) -> str:
    if c["status"] != "ok":
        return f"{GLYPH.get(c['status'], '?')} {c['status']}"
    a = c["assessment"]
    v = a.get("verdict", "?")
    if v == "at_reference":
        return f"✅ ≤ref ({a['scd']:.1f} digits)"
    return f"{GLYPH.get(v, '?')} {a['tol_ratio']:.3g}× ({a['scd']:.1f} digits)"


def build_report(results_path: str | Path, out_path: str | Path, references_path: str | Path | None = None) -> str:
    payload = json.loads(Path(results_path).read_text())
    meta, cases = payload["meta"], payload["cases"]
    refs = {}
    rp = Path(references_path) if references_path else Path(results_path).with_name("references.json")
    if rp.exists():
        refs = {r["problem"]: r for r in json.loads(rp.read_text())}
    solvers = sorted({c["solver"] for c in cases}, key=lambda s: (s.split(".")[0], s))
    rtols = meta["rtols"]
    by = defaultdict(dict)
    for c in cases:
        by[(c["problem"], c["solver"])][c["rtol"]] = c
    lines = [
        "# ODE solver accuracy audit",
        "",
        f"odeaudit {meta['odeaudit']} · Python {meta['python']} · {meta['platform']} · "
        + ", ".join(f"{k} {v}" for k, v in meta["libraries"].items()),
        "",
        f"Requested tolerances rtol ∈ {{{', '.join(f'{r:.0e}' for r in rtols)}}}, {meta['atol_rule']}. "
        f"Wall-clock budget per run {meta['timeout_s']:.0f} s.",
        "",
        "Each cell: *achieved error ÷ requested tolerance* in the solver's own mixed norm at the worst checkpoint, "
        "and the number of correct significant digits at the final time. ✅ ≤ 10× · ⚠️ 10–1000× · ❌ > 1000× while "
        "reporting success (a silent failure) · ⏱ did not finish in the budget · 🛑 solver reported failure · 💥 raised. "
        "“≤ref” means the error is below what the reference itself can resolve.",
        "",
    ]
    for group in dict.fromkeys(p.group for p in PROBLEMS.values()):
        probs = [p for p in PROBLEMS.values() if p.group == group and any(k[0] == p.name for k in by)]
        if not probs:
            continue
        lines += [f"## {group}", ""]
        for p in probs:
            r = refs.get(p.name, {})
            lines += [
                f"### {p.name} — {p.description}",
                f"t ∈ [{p.t_span[0]:g}, {p.t_span[1]:g}], dim {p.dim}, reference: {r.get('source', '?')}"
                + (f" (uncertainty {r.get('uncertainty', 0):.0e})" if r else ""),
                "",
                "| solver | " + " | ".join(f"rtol {rt:.0e}" for rt in rtols) + " |",
                "|---|" + "---|" * len(rtols),
            ]
            for s in solvers:
                row = by.get((p.name, s))
                if not row:
                    continue
                lines.append(f"| `{s}` | " + " | ".join(_cell(row[rt]) if rt in row else "·" for rt in rtols) + " |")
            lines.append("")
    # headline tables
    silent = [c for c in cases if c["status"] == "ok" and c["assessment"].get("verdict") == "silent_failure"]
    timeouts = [c for c in cases if c["status"] == "timeout"]
    failed = [c for c in cases if c["status"] in ("failed", "crashed")]
    lines += [
        "## Silent failures",
        "",
        "Solver reported success; the answer is more than 1000× worse than requested.",
        "",
    ]
    lines += ["| problem | solver | rtol | error/rtol | correct digits |", "|---|---|---|---|---|"]
    for c in sorted(silent, key=lambda c: -c["assessment"]["tol_ratio"]):
        a = c["assessment"]
        lines.append(f"| {c['problem']} | `{c['solver']}` | {c['rtol']:.0e} | {a['tol_ratio']:.3g} | {a['scd']:.1f} |")
    lines += ["", "## Did not finish within the budget", "", "| problem | solver | rtol |", "|---|---|---|"]
    for c in timeouts:
        lines.append(f"| {c['problem']} | `{c['solver']}` | {c['rtol']:.0e} |")
    lines += [
        "",
        "## Reported failures / crashes",
        "",
        "| problem | solver | rtol | status | message |",
        "|---|---|---|---|---|",
    ]
    for c in failed:
        lines.append(f"| {c['problem']} | `{c['solver']}` | {c['rtol']:.0e} | {c['status']} | {c['message'][:90]} |")
    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    return text
