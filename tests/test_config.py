from omegaconf import OmegaConf

from rl_recsys.config import ExperimentConfig, to_experiment_config


def test_to_experiment_config_merges_nested_values() -> None:
    raw = OmegaConf.create(
        {
            "env": {"slate_size": 7, "num_candidates": 25},
            "train": {"num_episodes": 12, "seed": 3},
        }
    )

    cfg = to_experiment_config(raw)

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.env.slate_size == 7
    assert cfg.env.num_candidates == 25
    assert cfg.train.num_episodes == 12
    assert cfg.train.seed == 3
    assert cfg.agent.name == "random"
    assert cfg.wandb.enabled is False
