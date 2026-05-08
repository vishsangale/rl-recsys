import numpy as np
import pandas as pd
import pytest
import torch

from rl_recsys.evaluation.behavior_policy import BehaviorPolicy


def test_slate_propensity_returns_product_of_position_softmax() -> None:
    # 3 candidate items, slate_size=2, feature_dim=2.
    # Construct a model and override its scorer to return known logits so
    # the softmax probabilities are hand-computable.
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )

    def fake_score(user_feat, candidate_feats, position):
        # Return position-dependent logits per candidate.
        if position == 0:
            return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

    model._score_position = fake_score  # monkey-patched scorer

    user = np.array([0.1, 0.2], dtype=np.float64)
    cand = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    slate = np.array([0, 1], dtype=np.int64)

    e = float(np.e)
    p_pos0_item0 = e / (e + 2.0)  # softmax([1,0,0])[0]
    p_pos1_item1 = e / (e + 2.0)  # softmax([0,1,0])[1]
    expected = p_pos0_item0 * p_pos1_item1

    result = model.slate_propensity(user, cand, slate)
    assert result == pytest.approx(expected)


def test_fit_behavior_policy_recovers_context_dependent_distribution(
    tmp_path,
) -> None:
    # Construct a synthetic logged dataset where context A always picks item 0
    # at position 0 and context B always picks item 1 at position 0. After
    # training, the model should put >0.5 probability on the correct item
    # given each context.
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        # Context A: user_state = [1, 0]
        rows.append({
            "session_id": rng.integers(1, 1000),
            "sequence_id": 1,
            "user_state": [1.0, 0.0],
            "slate": [0, 1, 2],
            "user_feedback": [1, 0, 0],
            "item_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
        })
        # Context B: user_state = [0, 1]
        rows.append({
            "session_id": rng.integers(1000, 2000),
            "sequence_id": 1,
            "user_state": [0.0, 1.0],
            "slate": [1, 0, 2],
            "user_feedback": [1, 0, 0],
            "item_features": [[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]],
        })
    df = pd.DataFrame(rows)
    parquet = tmp_path / "synth_b.parquet"
    df.to_parquet(parquet, index=False)

    from rl_recsys.evaluation.behavior_policy import fit_behavior_policy
    model = fit_behavior_policy(
        parquet, user_dim=2, item_dim=2, slate_size=3, num_items=3,
        epochs=20, batch_size=64, seed=0,
    )

    # In context A, position-0 prob for item 0 should exceed item 1's prob.
    user_a = np.array([1.0, 0.0], dtype=np.float64)
    cand = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    p_a_item0 = model.slate_propensity(user_a, cand, np.array([0], dtype=np.int64))
    p_a_item1 = model.slate_propensity(user_a, cand, np.array([1], dtype=np.int64))
    assert p_a_item0 > p_a_item1

    user_b = np.array([0.0, 1.0], dtype=np.float64)
    p_b_item0 = model.slate_propensity(user_b, cand, np.array([0], dtype=np.int64))
    p_b_item1 = model.slate_propensity(user_b, cand, np.array([1], dtype=np.int64))
    assert p_b_item1 > p_b_item0


def test_held_out_nll_returns_average_neg_log_prob(tmp_path) -> None:
    from rl_recsys.evaluation.behavior_policy import (
        BehaviorPolicy, held_out_nll,
    )
    # Build a model with a deterministic scorer for which NLL is hand-computable.
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=1, num_items=3,
        hidden_dim=4, seed=0,
    )
    # Patch _score_batch (used by the vectorized held_out_nll) to return
    # known logits: shape (B, 3) with [1, 0, 0] for every sample.
    def fake_score_batch(users, cands, positions):
        b = users.shape[0]
        return torch.tensor([[1.0, 0.0, 0.0]] * b, dtype=torch.float64)
    model._score_batch = fake_score_batch

    # Pass universe explicitly — df no longer carries candidate_features/candidate_ids.
    universe_ids = np.array([0, 1, 2], dtype=np.int64)
    universe_features = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    df = pd.DataFrame([
        {"user_state": [0.1, 0.2],
         "slate": [0]},  # target = position 0, item id 0
    ])
    e = float(np.e)
    expected = -np.log(e / (e + 2.0))  # softmax([1,0,0])[0]

    result = held_out_nll(
        model, df, universe_ids=universe_ids, universe_features=universe_features,
    )
    assert result == pytest.approx(expected, rel=1e-6)


def test_fit_behavior_policy_with_calibration_raises_on_bad_nll(
    tmp_path, monkeypatch,
) -> None:
    from rl_recsys.evaluation import behavior_policy as bp_module

    # Force fit_behavior_policy to return a degenerate model whose NLL on
    # held-out exceeds 2*log(num_items). We monkey-patch fit_behavior_policy
    # to a stub that returns a model with random weights only (no training).
    def stub_fit(*args, **kwargs):
        return bp_module.BehaviorPolicy(
            user_dim=kwargs["user_dim"], item_dim=kwargs["item_dim"],
            slate_size=kwargs["slate_size"], num_items=kwargs["num_items"],
            hidden_dim=4, seed=0,
        )
    monkeypatch.setattr(bp_module, "fit_behavior_policy", stub_fit)

    _item_feats = {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.5, 0.5]}
    rows = [
        {"session_id": i, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [(i % 3)], "user_feedback": [1],
         "item_features": [_item_feats[i % 3]]}
        for i in range(50)
    ]
    df = pd.DataFrame(rows)
    parquet = tmp_path / "noisy_b.parquet"
    df.to_parquet(parquet, index=False)

    # NLL threshold gate: with the stub returning an untrained model, NLL on
    # noisy held-out should exceed 2*log(3) ≈ 2.197 only if the model is
    # severely biased. To FORCE a fail, set threshold ratio to a tiny value.
    with pytest.raises(ValueError, match="behavior policy NLL exceeds threshold"):
        bp_module.fit_behavior_policy_with_calibration(
            parquet, user_dim=2, item_dim=2, slate_size=1, num_items=3,
            epochs=1, batch_size=8, seed=0, nll_threshold=0.01,
        )
