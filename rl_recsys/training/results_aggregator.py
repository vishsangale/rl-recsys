from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def aggregate(results_dir: Path) -> pd.DataFrame:
    """Read every {agent}_seed{seed}_pretrained{0|1}.json (skipping
    *.failed.json) into a long-form DataFrame with one row per run."""
    rows: list[dict] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        data = json.loads(path.read_text())
        flat = {
            "agent": data["agent"],
            "seed": int(data["seed"]),
            "pretrained": bool(data["pretrained"]),
        }
        for k, v in data.get("metrics", {}).items():
            flat[k] = v
        flat["git_sha"] = data.get("git_sha", "unknown")
        flat["timestamp"] = data.get("timestamp", "")
        rows.append(flat)
    return pd.DataFrame(rows)


def to_summary_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def to_summary_md(df: pd.DataFrame) -> str:
    """Pivot table: rows=agent, cols=pretrained ∈ {True, False}, cells = mean ± std
    of avg_seq_dr_value across seeds."""
    grouped = df.groupby(["agent", "pretrained"])["avg_seq_dr_value"].agg(
        ["mean", "std"]
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    cells = grouped.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1,
    )
    pivot = cells.unstack("pretrained")
    if True in pivot.columns:
        pivot = pivot.sort_values(by=True, ascending=False)

    headers = ["agent"] + [f"pretrained={c}" for c in pivot.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for agent, row in pivot.iterrows():
        cells_row = [agent] + [str(v) if pd.notna(v) else "—" for v in row]
        lines.append("| " + " | ".join(cells_row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1])
    df = aggregate(target)
    to_summary_csv(df, target / "summary.csv")
    (target / "summary.md").write_text(to_summary_md(df))
    print(f"Wrote {target/'summary.csv'} and {target/'summary.md'}")
