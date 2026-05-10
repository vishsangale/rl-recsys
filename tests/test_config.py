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
    assert cfg.wandb.base_url is None
    assert cfg.mlflow.enabled is False
    assert cfg.mlflow.tracking_uri == "sqlite:///mlflow.db"
    assert cfg.runtime.project_name == "rl-recsys"
    assert cfg.runtime.results_root.endswith("/results")


def test_agent_config_supports_all_registered_agents():
    from rl_recsys.agents.factory import AGENT_REGISTRY, build_agent
    from rl_recsys.config import AgentConfig, EnvConfig

    env = EnvConfig(slate_size=3, user_dim=4, item_dim=3, num_candidates=8)
    for name in AGENT_REGISTRY:
        cfg = AgentConfig(name=name)
        agent = build_agent(cfg, env)
        assert agent is not None, f"factory returned None for agent {name!r}"
