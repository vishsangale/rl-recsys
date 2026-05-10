from __future__ import annotations

import json
from pathlib import Path

from rl_recsys.training.results_aggregator import (
    aggregate, to_summary_csv, to_summary_md,
)


def _write(dirpath: Path, payload: dict, name: str) -> None:
    (dirpath / name).write_text(json.dumps(payload))


def test_aggregate_reads_all_json_skips_failed(tmp_path):
    _write(tmp_path, {
        "agent": "linucb", "seed": 0, "pretrained": True,
        "metrics": {"avg_seq_dr_value": 10.0, "avg_logged_discounted_return": 9.0},
        "config": {}, "git_sha": "x", "timestamp": "2026-05-09T00:00:00Z",
    }, "linucb_seed0_pretrained1.json")
    _write(tmp_path, {
        "agent": "broken", "seed": 0, "pretrained": True,
        "error": "boom", "traceback": "...",
    }, "broken_seed0_pretrained1.failed.json")
    df = aggregate(tmp_path)
    assert len(df) == 1
    assert df.loc[0, "agent"] == "linucb"


def test_summary_md_pivots_pretrained_columns(tmp_path):
    for seed in (0, 1, 2):
        _write(tmp_path, {
            "agent": "linucb", "seed": seed, "pretrained": True,
            "metrics": {"avg_seq_dr_value": 10.0 + seed * 0.1,
                        "avg_logged_discounted_return": 9.0},
            "config": {}, "git_sha": "x", "timestamp": "t",
        }, f"linucb_seed{seed}_pretrained1.json")
        _write(tmp_path, {
            "agent": "linucb", "seed": seed, "pretrained": False,
            "metrics": {"avg_seq_dr_value": 9.5 + seed * 0.05,
                        "avg_logged_discounted_return": 9.0},
            "config": {}, "git_sha": "x", "timestamp": "t",
        }, f"linucb_seed{seed}_pretrained0.json")
    df = aggregate(tmp_path)
    md = to_summary_md(df)
    assert "linucb" in md
    assert "True" in md and "False" in md
