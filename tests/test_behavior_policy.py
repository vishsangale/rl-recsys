import numpy as np
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
