from __future__ import annotations

import json
import subprocess
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

from rl_recsys.agents.factory import build_agent
from rl_recsys.config import AgentConfig, EnvConfig
from rl_recsys.evaluation.ope_trajectory import (
    LoggedTrajectorySource,
    evaluate_trajectory_ope_agent,
)


@dataclass
class GridRun:
    agent_name: str
    seed: int
    pretrained: bool
    config_overrides: dict = field(default_factory=dict)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _run_id(run: GridRun) -> str:
    return f"{run.agent_name}_seed{run.seed}_pretrained{int(run.pretrained)}"


def run_grid(
    runs: list[GridRun],
    *,
    train_source_factory: Callable[[int], LoggedTrajectorySource],
    eval_source_factory: Callable[[int], LoggedTrajectorySource],
    env_kwargs: dict,
    output_dir: Path,
    max_trajectories: int,
    boltzmann_T: float = 1.0,
    resume: bool = False,
    behavior_policy=None,
) -> list[Path]:
    """Run each (agent, seed, pretrained) tuple, write one JSON per run.

    Returns the list of paths written (excluding skipped-on-resume runs).
    Failed runs write a {run_id}.failed.json with the exception trace and
    do not count toward `written`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    env_cfg = EnvConfig(**env_kwargs)

    for run in runs:
        rid = _run_id(run)
        out_path = output_dir / f"{rid}.json"
        if resume and out_path.exists():
            continue

        agent_cfg = AgentConfig(name=run.agent_name)
        for k, v in run.config_overrides.items():
            setattr(agent_cfg, k, v)

        try:
            agent = build_agent(agent_cfg, env_cfg)
            train_src = train_source_factory(run.seed)
            eval_src = eval_source_factory(run.seed)

            # BC/GBDT need the real candidate_features from the source
            # before training. The factory builds them with placeholders
            # because it can't see the source.
            # Guard: only overwrite when the agent stores _candidate_features
            # as a data attribute (ndarray), NOT when it's a bound method
            # (e.g. LinUCBAgent._candidate_features computes features on the fly).
            if (
                hasattr(agent, "_candidate_features")
                and not callable(agent._candidate_features)
                and hasattr(train_src, "_candidate_features")
            ):
                agent._candidate_features = train_src._candidate_features
            # BC also needs an injected, pre-fit BehaviorPolicy.
            if behavior_policy is not None and hasattr(
                agent, "inject_behavior_policy",
            ):
                agent.inject_behavior_policy(behavior_policy)

            train_started = perf_counter()
            if run.pretrained:
                agent.train_offline(train_src, seed=run.seed)
            train_seconds = perf_counter() - train_started

            evaluation = evaluate_trajectory_ope_agent(
                eval_src, agent,
                agent_name=run.agent_name,
                max_trajectories=max_trajectories,
                seed=run.seed,
                temperature=boltzmann_T,
            )
            payload = {
                "agent": run.agent_name,
                "seed": run.seed,
                "pretrained": run.pretrained,
                "config": {**vars(agent_cfg), "boltzmann_T": boltzmann_T},
                "metrics": {
                    **evaluation.as_dict(),
                    "train_seconds": train_seconds,
                },
                "git_sha": _git_sha(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out_path.write_text(json.dumps(payload, indent=2, default=float))
            written.append(out_path)
        except Exception as exc:
            failed_path = output_dir / f"{rid}.failed.json"
            failed_path.write_text(json.dumps({
                "agent": run.agent_name,
                "seed": run.seed,
                "pretrained": run.pretrained,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }, indent=2))
            print(f"[grid] {rid} FAILED: {exc!r}", flush=True)
    return written
