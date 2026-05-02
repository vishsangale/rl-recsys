# Dataset-Backed RL Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add KuaiRecEnv, FinnNoSlateEnv, and RL4RSEnv — three RecEnv subclasses backed by real logged datasets — through two thin shared base classes (BanditDatasetEnv, SessionDatasetEnv) and a shared hashed_vector utility.

**Architecture:** Extract `_hashed_vector` from `logged.py` into a shared `features.py`; add `BanditDatasetEnv` (single-step bandit, sample positive + negatives each episode) and `SessionDatasetEnv` (multi-step, step through a logged session) to `dataset_base.py`; build KuaiRecEnv, FinnNoSlateEnv, and RL4RSEnv on top of these; update the KuaiRec and RL4RS pipelines to emit item_features.parquet and sessions.parquet respectively.

**Tech Stack:** Python 3.10+, NumPy, Pandas, PyArrow. No new dependencies.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `rl_recsys/environments/features.py` | Create | `hashed_vector(prefix, entity_id, dim)` utility |
| `rl_recsys/environments/logged.py` | Modify | Import `hashed_vector` from features.py; remove local copy |
| `rl_recsys/environments/dataset_base.py` | Create | `BanditDatasetEnv`, `SessionDatasetEnv` abstract bases |
| `rl_recsys/environments/kuairec.py` | Create | `KuaiRecEnv(BanditDatasetEnv)` |
| `rl_recsys/environments/finn_no_slate.py` | Create | `FinnNoSlateEnv(BanditDatasetEnv)` |
| `rl_recsys/environments/rl4rs.py` | Create | `RL4RSEnv(SessionDatasetEnv)` |
| `rl_recsys/environments/__init__.py` | Modify | Export new env classes |
| `rl_recsys/data/pipelines/kuairec.py` | Modify | Emit `item_features.parquet` |
| `rl_recsys/data/pipelines/rl4rs.py` | Modify | Replace placeholder; emit `sessions.parquet` |
| `tests/test_features.py` | Create | Tests for `hashed_vector` |
| `tests/test_dataset_base.py` | Create | Tests for both base classes |
| `tests/test_env_kuairec.py` | Create | Tests for `KuaiRecEnv` |
| `tests/test_env_finn_no_slate.py` | Create | Tests for `FinnNoSlateEnv` |
| `tests/test_env_rl4rs.py` | Create | Tests for `RL4RSEnv` |
| `tests/test_pipeline_rl4rs.py` | Create | Tests for updated RL4RS pipeline |
| `tests/test_pipeline_kuairec.py` | Modify | Add test for item_features.parquet |

---

### Task 1: Extract hashed_vector to shared features.py

**Files:**
- Create: `rl_recsys/environments/features.py`
- Modify: `rl_recsys/environments/logged.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import numpy as np
from rl_recsys.environments.features import hashed_vector


def test_hashed_vector_shape():
    v = hashed_vector("user", 42, 16)
    assert v.shape == (16,)


def test_hashed_vector_is_unit_norm():
    v = hashed_vector("item", 7, 8)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_hashed_vector_deterministic():
    v1 = hashed_vector("user", 1, 16)
    v2 = hashed_vector("user", 1, 16)
    np.testing.assert_array_equal(v1, v2)


def test_hashed_vector_differs_by_prefix():
    v1 = hashed_vector("user", 1, 16)
    v2 = hashed_vector("item", 1, 16)
    assert not np.allclose(v1, v2)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_features.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.environments.features'`

- [ ] **Step 3: Create `rl_recsys/environments/features.py`**

```python
from __future__ import annotations

import hashlib

import numpy as np


def hashed_vector(prefix: str, entity_id: int, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(
        f"{prefix}:{entity_id}".encode("utf-8"), digest_size=8
    )
    seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_features.py -v
```
Expected: 4 passed

- [ ] **Step 5: Update `rl_recsys/environments/logged.py`**

Replace line 3 (`import hashlib`) with:
```python
from rl_recsys.environments.features import hashed_vector
```

Replace lines 153–161 (the entire `_hashed_vector` function):
```python
def _hashed_vector(prefix: str, entity_id: int, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(f"{prefix}:{entity_id}".encode("utf-8"), digest_size=8)
    seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
```
with a one-liner alias so the call site in `_build_entity_features` doesn't change:
```python
_hashed_vector = hashed_vector
```

- [ ] **Step 6: Verify existing tests still pass**

```bash
pytest tests/test_logged_bandit.py tests/test_environments.py -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/environments/features.py rl_recsys/environments/logged.py tests/test_features.py
git commit -m "refactor: extract hashed_vector to shared features.py"
```

---

### Task 2: BanditDatasetEnv base class

**Files:**
- Create: `rl_recsys/environments/dataset_base.py`
- Create: `tests/test_dataset_base.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset_base.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.base import RecObs, RecStep
from rl_recsys.environments.dataset_base import BanditDatasetEnv


class _SimpleBanditEnv(BanditDatasetEnv):
    def _compute_reward(self, row, clicks):
        return float(clicks.sum())


def _interactions(n_users=5, n_items=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {"user_id": u, "item_id": i, "rating": float(rng.uniform(0, 1))}
        for u in range(n_users)
        for i in range(n_items)
    ]
    return pd.DataFrame(rows)


def test_bandit_reset_obs_shape():
    env = _SimpleBanditEnv(_interactions(), slate_size=2, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    obs = env.reset(seed=42)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (10, 8)
    assert obs.candidate_ids.shape == (10,)


def test_bandit_step_done_true():
    env = _SimpleBanditEnv(_interactions(), slate_size=2, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1]))
    assert step.done is True
    assert step.clicks.shape == (2,)
    assert isinstance(step.reward, float)


def test_bandit_step_before_reset_raises():
    env = _SimpleBanditEnv(_interactions(), num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0]))


def test_bandit_empty_df_raises():
    with pytest.raises(ValueError, match="empty"):
        _SimpleBanditEnv(pd.DataFrame(columns=["user_id", "item_id"]), num_candidates=5, feature_dim=4, feature_source="hashed")


def test_bandit_too_many_candidates_raises():
    with pytest.raises(ValueError, match="num_candidates"):
        _SimpleBanditEnv(_interactions(n_items=5), num_candidates=10, feature_dim=4, feature_source="hashed")


def test_bandit_slate_exceeds_candidates_raises():
    with pytest.raises(ValueError, match="slate_size"):
        _SimpleBanditEnv(_interactions(), slate_size=15, num_candidates=10, feature_dim=4, feature_source="hashed")


def test_bandit_feature_dim_too_small_raises():
    with pytest.raises(ValueError, match="feature_dim"):
        _SimpleBanditEnv(_interactions(), num_candidates=10, feature_dim=2, feature_source="hashed")


def test_bandit_properties():
    env = _SimpleBanditEnv(_interactions(), slate_size=3, num_candidates=10, feature_dim=8, feature_source="hashed")
    assert env.slate_size == 3
    assert env.num_candidates == 10
    assert env.user_dim == 8
    assert env.item_dim == 8


def test_bandit_positive_item_in_candidates():
    env = _SimpleBanditEnv(_interactions(), slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=7)
    obs = env.reset(seed=7)
    # Selecting all candidates must hit the positive item at least once
    step = env.step(np.arange(10))
    assert step.clicks.sum() >= 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_dataset_base.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.environments.dataset_base'`

- [ ] **Step 3: Create `rl_recsys/environments/dataset_base.py` with `BanditDatasetEnv`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from rl_recsys.environments.base import RecEnv, RecObs, RecStep
from rl_recsys.environments.features import hashed_vector


class BanditDatasetEnv(RecEnv, ABC):
    """Single-step bandit environment from logged interaction data.

    Each episode: sample one interaction row, build a candidate pool of
    num_candidates items (positive + sampled negatives), return done=True
    after one step.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        *,
        slate_size: int = 1,
        num_candidates: int = 50,
        feature_dim: int = 16,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        if interactions.empty:
            raise ValueError("interactions DataFrame is empty")
        missing = {"user_id", "item_id"} - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing columns: {sorted(missing)}")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")
        n_items = interactions["item_id"].nunique()
        if num_candidates > n_items:
            raise ValueError(
                f"num_candidates={num_candidates} exceeds unique item count={n_items}"
            )
        if slate_size > num_candidates:
            raise ValueError("slate_size must be <= num_candidates")
        if feature_source not in ("native", "hashed"):
            raise ValueError(
                f"feature_source must be 'native' or 'hashed', got {feature_source!r}"
            )

        self._df = interactions.reset_index(drop=True)
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._feature_dim = feature_dim
        self._feature_source = feature_source
        self._rng = np.random.default_rng(seed)
        self._all_items = np.array(
            sorted(interactions["item_id"].unique()), dtype=np.int64
        )
        self._user_positive_items: dict[int, set[int]] = {
            int(uid): set(grp["item_id"].astype(int))
            for uid, grp in interactions.groupby("user_id")
        }
        self._current_row: pd.Series | None = None
        self._current_candidate_ids: np.ndarray | None = None

    @property
    def slate_size(self) -> int:
        return self._slate_size

    @property
    def num_candidates(self) -> int:
        return self._num_candidates

    @property
    def user_dim(self) -> int:
        return self._feature_dim

    @property
    def item_dim(self) -> int:
        return self._feature_dim

    def reset(self, seed: int | None = None) -> RecObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, len(self._df)))
        self._current_row = self._df.iloc[idx]
        positive_id = int(self._current_row["item_id"])
        user_id = int(self._current_row["user_id"])
        self._current_candidate_ids = self._build_candidate_ids(
            self._current_row, positive_id, user_id
        )
        return RecObs(
            user_features=self._get_user_features(self._current_row).astype(np.float32),
            candidate_features=self._get_item_features(
                self._current_row, self._current_candidate_ids
            ).astype(np.float32),
            candidate_ids=self._current_candidate_ids.copy(),
        )

    def step(self, slate: np.ndarray) -> RecStep:
        if self._current_row is None or self._current_candidate_ids is None:
            raise RuntimeError("reset() must be called before step()")
        slate = np.asarray(slate, dtype=np.int64)
        selected_ids = self._current_candidate_ids[slate]
        positive_id = int(self._current_row["item_id"])
        clicks = (selected_ids == positive_id).astype(np.float32)
        reward = self._compute_reward(self._current_row, clicks)
        obs = RecObs(
            user_features=self._get_user_features(self._current_row).astype(np.float32),
            candidate_features=self._get_item_features(
                self._current_row, self._current_candidate_ids
            ).astype(np.float32),
            candidate_ids=self._current_candidate_ids.copy(),
        )
        return RecStep(obs=obs, reward=reward, clicks=clicks, done=True)

    def _build_candidate_ids(
        self, row: pd.Series, positive_id: int, user_id: int
    ) -> np.ndarray:
        excluded = self._user_positive_items.get(user_id, set()) | {positive_id}
        pool = np.array(
            [iid for iid in self._all_items if int(iid) not in excluded],
            dtype=np.int64,
        )
        needed = self._num_candidates - 1
        if len(pool) < needed:
            pool = self._all_items[self._all_items != positive_id]
        negatives = self._rng.choice(pool, size=needed, replace=False)
        candidate_ids = np.concatenate([[positive_id], negatives])
        self._rng.shuffle(candidate_ids)
        return candidate_ids

    def _get_user_features(self, row: pd.Series) -> np.ndarray:
        return hashed_vector("user", int(row["user_id"]), self._feature_dim)

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        return np.stack(
            [hashed_vector("item", int(iid), self._feature_dim) for iid in candidate_ids]
        ).astype(np.float32)

    @abstractmethod
    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float: ...
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_dataset_base.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/environments/dataset_base.py tests/test_dataset_base.py
git commit -m "feat: add BanditDatasetEnv base class"
```

---

### Task 3: SessionDatasetEnv base class

**Files:**
- Modify: `rl_recsys/environments/dataset_base.py` (append `SessionDatasetEnv`)
- Modify: `tests/test_dataset_base.py` (append session tests)

- [ ] **Step 1: Write the failing tests (append to `tests/test_dataset_base.py`)**

```python
from rl_recsys.environments.dataset_base import SessionDatasetEnv


class _SimpleSessionEnv(SessionDatasetEnv):
    def _compute_reward(self, row, clicks):
        return float(clicks.sum())


def _sessions(n_sessions=2, n_steps=3, slate_size=3) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(1)
    result = {}
    for sid in range(n_sessions):
        rows = []
        for _ in range(n_steps):
            items = rng.integers(0, 100, size=slate_size).tolist()
            feats = rng.standard_normal((slate_size, 4)).tolist()
            clicks_vec = [0] * slate_size
            clicks_vec[int(rng.integers(0, slate_size))] = 1
            rows.append({
                "slate": [int(x) for x in items],
                "item_features": feats,
                "clicks": clicks_vec,
                "user_state": rng.standard_normal(4).tolist(),
            })
        result[sid] = pd.DataFrame(rows)
    return result


def test_session_reset_obs_shape():
    env = _SimpleSessionEnv(_sessions(), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (4,)
    assert obs.candidate_features.shape == (3, 4)
    assert obs.candidate_ids.shape == (3,)


def test_session_done_false_mid_session():
    env = _SimpleSessionEnv(_sessions(), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1, 2]))
    assert step.done is False


def test_session_done_true_at_end():
    env = _SimpleSessionEnv(_sessions(n_steps=3), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    env.step(np.array([0, 1, 2]))
    env.step(np.array([0, 1, 2]))
    step = env.step(np.array([0, 1, 2]))
    assert step.done is True


def test_session_step_before_reset_raises():
    env = _SimpleSessionEnv(_sessions(), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0, 1, 2]))


def test_session_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        _SimpleSessionEnv({}, slate_size=1, num_candidates=1, feature_dim=4, feature_source="hashed")


def test_session_next_obs_shape_mid_session():
    env = _SimpleSessionEnv(_sessions(n_steps=3), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1, 2]))
    assert step.obs.user_features.shape == (4,)
    assert step.obs.candidate_features.shape == (3, 4)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_dataset_base.py::test_session_reset_obs_shape -v
```
Expected: `ImportError: cannot import name 'SessionDatasetEnv'`

- [ ] **Step 3: Append `SessionDatasetEnv` to `rl_recsys/environments/dataset_base.py`**

```python
class SessionDatasetEnv(RecEnv, ABC):
    """Multi-step session environment from logged session data.

    Each episode samples one session (dict[session_id -> DataFrame]).
    Steps through session rows until done=True at the final step.
    Candidate pool = the logged slate at each step (num_candidates fixed to slate length).
    """

    def __init__(
        self,
        sessions: dict[int, pd.DataFrame],
        *,
        slate_size: int = 1,
        num_candidates: int = 50,
        feature_dim: int = 16,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        if not sessions:
            raise ValueError("sessions dict is empty")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")
        if feature_source not in ("native", "hashed"):
            raise ValueError(
                f"feature_source must be 'native' or 'hashed', got {feature_source!r}"
            )

        self._sessions = sessions
        self._session_ids = list(sessions.keys())
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._feature_dim = feature_dim
        self._feature_source = feature_source
        self._rng = np.random.default_rng(seed)
        self._current_session: pd.DataFrame | None = None
        self._current_session_id: int | None = None
        self._cursor: int = 0

    @property
    def slate_size(self) -> int:
        return self._slate_size

    @property
    def num_candidates(self) -> int:
        return self._num_candidates

    @property
    def user_dim(self) -> int:
        return self._feature_dim

    @property
    def item_dim(self) -> int:
        return self._feature_dim

    def reset(self, seed: int | None = None) -> RecObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, len(self._session_ids)))
        self._current_session_id = self._session_ids[idx]
        self._current_session = self._sessions[self._current_session_id].reset_index(drop=True)
        self._cursor = 0
        return self._obs_at_cursor()

    def step(self, slate: np.ndarray) -> RecStep:
        if self._current_session is None:
            raise RuntimeError("reset() must be called before step()")
        row = self._current_session.iloc[self._cursor]
        candidate_ids = np.array(row["slate"], dtype=np.int64)
        slate = np.asarray(slate, dtype=np.int64)
        selected_ids = candidate_ids[slate]
        logged_clicks = np.array(row["clicks"], dtype=np.float32)
        clicked_ids = candidate_ids[logged_clicks > 0]
        clicks = np.isin(selected_ids, clicked_ids).astype(np.float32)
        reward = self._compute_reward(row, clicks)
        self._cursor += 1
        done = self._cursor >= len(self._current_session)
        next_obs = self._obs_at_cursor() if not done else self._obs_for_row(row, candidate_ids)
        return RecStep(obs=next_obs, reward=reward, clicks=clicks, done=done)

    def _obs_at_cursor(self) -> RecObs:
        row = self._current_session.iloc[self._cursor]
        candidate_ids = np.array(row["slate"], dtype=np.int64)
        return self._obs_for_row(row, candidate_ids)

    def _obs_for_row(self, row: pd.Series, candidate_ids: np.ndarray) -> RecObs:
        return RecObs(
            user_features=self._get_user_features(row).astype(np.float32),
            candidate_features=self._get_item_features(row, candidate_ids).astype(np.float32),
            candidate_ids=candidate_ids.copy(),
        )

    def _get_user_features(self, row: pd.Series) -> np.ndarray:
        sid = self._current_session_id if self._current_session_id is not None else -1
        return hashed_vector("session", sid, self._feature_dim)

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        return np.stack(
            [hashed_vector("item", int(iid), self._feature_dim) for iid in candidate_ids]
        ).astype(np.float32)

    @abstractmethod
    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float: ...
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_dataset_base.py -v
```
Expected: all 15 tests pass

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/environments/dataset_base.py tests/test_dataset_base.py
git commit -m "feat: add SessionDatasetEnv base class"
```

---

### Task 4: Update KuaiRec pipeline to emit item_features.parquet

**Files:**
- Modify: `rl_recsys/data/pipelines/kuairec.py`
- Modify: `tests/test_pipeline_kuairec.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_pipeline_kuairec.py`)**

```python
def test_process_emits_item_features_parquet(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "KuaiRec 2.0" / "data"
    extracted.mkdir(parents=True)
    (extracted / "big_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,0.5\n"
        "1,101,45.0,90.0,1609459260,2021-01-01,0.5\n"
    )
    (extracted / "item_categories.csv").write_text(
        "video_id,feat\n"
        '100,"[1, 2]"\n'
        '101,"[3, 4]"\n'
    )
    proc_dir = tmp_path / "proc"
    p = KuaiRecPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "item_features.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert "item_id" in df.columns
    assert len(df) == 2
    feat_cols = [c for c in df.columns if c != "item_id"]
    assert len(feat_cols) > 0


def test_process_skips_item_features_when_categories_absent(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "KuaiRec 2.0" / "data"
    extracted.mkdir(parents=True)
    (extracted / "big_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,0.5\n"
    )
    # no item_categories.csv — should not raise
    proc_dir = tmp_path / "proc"
    p = KuaiRecPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()
    assert not (proc_dir / "item_features.parquet").exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_pipeline_kuairec.py::test_process_emits_item_features_parquet -v
```
Expected: `AssertionError` (file does not exist)

- [ ] **Step 3: Update `rl_recsys/data/pipelines/kuairec.py`**

Add `import ast` to the imports at the top of the file. Replace the existing `process` method with:

```python
    def process(self) -> None:
        matrix_file = self.raw_dir / "KuaiRec 2.0" / "data" / "big_matrix.csv"
        if not matrix_file.exists():
            raise FileNotFoundError(f"Not found: {matrix_file}. Run --download first.")

        df = pd.read_csv(matrix_file)
        df = df.rename(
            columns={
                "video_id": "item_id",
                "watch_ratio": "rating",
                "time": "timestamp",
            }
        )

        out = self.processed_dir / "interactions.parquet"
        df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved {len(df):,} rows to {out}")

        cats_file = self.raw_dir / "KuaiRec 2.0" / "data" / "item_categories.csv"
        if cats_file.exists():
            self._process_item_features(cats_file)
        else:
            print(f"item_categories.csv not found at {cats_file}; skipping item_features.parquet")

    def _process_item_features(self, cats_file: Path) -> None:
        import ast
        cats = pd.read_csv(cats_file).rename(columns={"video_id": "item_id"})
        if "feat" in cats.columns:
            cats["feat"] = cats["feat"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            all_cats = sorted({c for feats in cats["feat"] for c in feats})
            for cat in all_cats:
                cats[f"cat_{cat}"] = cats["feat"].apply(lambda x: int(cat in x))
            cats = cats.drop(columns=["feat"])
        out = self.processed_dir / "item_features.parquet"
        cats.to_parquet(out, index=False)
        print(f"Saved {len(cats):,} item feature rows to {out}")
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_pipeline_kuairec.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/pipelines/kuairec.py tests/test_pipeline_kuairec.py
git commit -m "feat: emit item_features.parquet from KuaiRec pipeline"
```

---

### Task 5: KuaiRecEnv

**Files:**
- Create: `rl_recsys/environments/kuairec.py`
- Create: `tests/test_env_kuairec.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_env_kuairec.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.kuairec import KuaiRecEnv


def _interactions(n_users=5, n_items=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"user_id": u, "item_id": i, "rating": float(rng.uniform(0, 1)), "timestamp": u * 1000 + i}
        for u in range(n_users) for i in range(n_items)
    ])


def _item_features(n_items=20) -> pd.DataFrame:
    return pd.DataFrame([
        {"item_id": i, "cat_0": i % 2, "cat_1": (i // 2) % 2}
        for i in range(n_items)
    ])


def test_reset_obs_shape(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    obs = env.reset(seed=42)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (10, 8)
    assert obs.candidate_ids.shape == (10,)


def test_step_done_true(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0]))
    assert step.done is True


def test_reward_in_zero_one(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    for seed in range(20):
        env.reset(seed=seed)
        step = env.step(np.arange(10))
        assert 0.0 <= step.reward <= 1.0


def test_native_raises_without_item_features(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    with pytest.raises(FileNotFoundError, match="item_features.parquet"):
        KuaiRecEnv(tmp_path, feature_source="native", num_candidates=10, feature_dim=8)


def test_native_candidate_feature_shape(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    _item_features().to_parquet(tmp_path / "item_features.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="native", seed=0)
    obs = env.reset(seed=0)
    assert obs.candidate_features.shape == (10, 8)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_env_kuairec.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.environments.kuairec'`

- [ ] **Step 3: Create `rl_recsys/environments/kuairec.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.environments.dataset_base import BanditDatasetEnv
from rl_recsys.environments.features import hashed_vector


class KuaiRecEnv(BanditDatasetEnv):
    """Bandit environment backed by KuaiRec logged interactions.

    Reward = watch_ratio * click (continuous, 0–1).
    feature_source="native": item features from item_features.parquet (category
    one-hot vectors); user features always hashed (KuaiRec has no user metadata).
    feature_source="hashed": all features hashed from entity IDs.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        slate_size: int = 1,
        num_candidates: int = 50,
        feature_dim: int = 16,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        processed_dir = Path(processed_dir)
        interactions = pd.read_parquet(processed_dir / "interactions.parquet")

        self._native_item_feat_map: dict[int, np.ndarray] | None = None
        if feature_source == "native":
            feat_path = processed_dir / "item_features.parquet"
            if not feat_path.exists():
                raise FileNotFoundError(
                    f"item_features.parquet not found at {feat_path}. "
                    "Rerun pipeline with --process to generate it."
                )
            feat_df = pd.read_parquet(feat_path)
            feat_cols = [c for c in feat_df.columns if c != "item_id"]
            n_feat = len(feat_cols)
            self._native_item_feat_map = {}
            for _, row in feat_df.iterrows():
                raw = row[feat_cols].to_numpy(dtype=np.float32)
                if n_feat < feature_dim:
                    padded = np.zeros(feature_dim, dtype=np.float32)
                    padded[:n_feat] = raw
                else:
                    padded = raw[:feature_dim]
                self._native_item_feat_map[int(row["item_id"])] = padded

        super().__init__(
            interactions,
            slate_size=slate_size,
            num_candidates=num_candidates,
            feature_dim=feature_dim,
            feature_source=feature_source,
            seed=seed,
        )

    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float:
        return float(row["rating"]) * float(clicks.sum())

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        if self._feature_source == "native" and self._native_item_feat_map is not None:
            vecs = []
            for iid in candidate_ids:
                feat = self._native_item_feat_map.get(int(iid))
                if feat is None:
                    feat = hashed_vector("item", int(iid), self._feature_dim).astype(np.float32)
                vecs.append(feat)
            return np.stack(vecs)
        return super()._get_item_features(row, candidate_ids)
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_env_kuairec.py -v
```
Expected: 5 passed

- [ ] **Step 5: Run full suite**

```bash
pytest -v --tb=short
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/environments/kuairec.py tests/test_env_kuairec.py
git commit -m "feat: add KuaiRecEnv"
```

---

### Task 6: FinnNoSlateEnv

**Files:**
- Create: `rl_recsys/environments/finn_no_slate.py`
- Create: `tests/test_env_finn_no_slate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_env_finn_no_slate.py
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.finn_no_slate import FinnNoSlateEnv


def _slates(n=30) -> pd.DataFrame:
    # n=30 so unique clicked items (30) >= num_candidates (25) for base class validation
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        slate = list(range(i * 25, i * 25 + 25))  # non-overlapping slates
        click_idx = int(rng.integers(0, 25))
        rows.append({
            "request_id": i,
            "user_id": i % 5,
            "slate": slate,
            "clicks": click_idx,
            "timestamp": 1600000000 + i * 1000,
        })
    return pd.DataFrame(rows)


def test_reset_obs_shape(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=5, feature_dim=8, seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (25, 8)
    assert obs.candidate_ids.shape == (25,)


def test_candidate_ids_match_a_logged_slate(tmp_path):
    df = _slates()
    df.to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=1, feature_dim=8, seed=0)
    for _ in range(5):
        obs = env.reset()
        candidate_set = set(obs.candidate_ids.tolist())
        found = any(
            set(row["slate"]) == candidate_set
            for _, row in df.iterrows()
        )
        assert found, "candidate_ids do not match any logged slate"


def test_step_done_true(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=1, feature_dim=8, seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0]))
    assert step.done is True


def test_reward_binary(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=25, feature_dim=8, seed=0)
    for seed in range(10):
        env.reset(seed=seed)
        step = env.step(np.arange(25))
        assert step.reward in (0.0, 1.0)


def test_native_fallback_warns(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FinnNoSlateEnv(tmp_path, feature_dim=8, feature_source="native", seed=0)
    assert any("hashed" in str(x.message).lower() or "native" in str(x.message).lower() for x in w)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_env_finn_no_slate.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.environments.finn_no_slate'`

- [ ] **Step 3: Create `rl_recsys/environments/finn_no_slate.py`**

```python
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.environments.dataset_base import BanditDatasetEnv


class FinnNoSlateEnv(BanditDatasetEnv):
    """Bandit environment backed by FINN.no Slate logged impressions.

    Each episode uses the actual logged 25-item slate as the candidate pool,
    preserving the real impression distribution. num_candidates is always 25.
    Reward is binary (1.0 if the agent selects the logged click, else 0.0).
    FINN.no has no item metadata: native feature_source falls back to hashed.
    """

    _LOGGED_SLATE_SIZE = 25

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        slate_size: int = 25,
        feature_dim: int = 16,
        feature_source: str = "hashed",
        seed: int = 0,
    ) -> None:
        processed_dir = Path(processed_dir)
        df = pd.read_parquet(processed_dir / "slates.parquet")

        if feature_source == "native":
            warnings.warn(
                "FinnNoSlateEnv: feature_source='native' is not supported "
                "(FINN.no has no item metadata); falling back to 'hashed'.",
                UserWarning,
                stacklevel=2,
            )
            feature_source = "hashed"

        df = df.copy()
        df["item_id"] = df.apply(lambda r: r["slate"][int(r["clicks"])], axis=1)
        df["rating"] = 1.0

        super().__init__(
            df,
            slate_size=slate_size,
            num_candidates=self._LOGGED_SLATE_SIZE,
            feature_dim=feature_dim,
            feature_source=feature_source,
            seed=seed,
        )

    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float:
        return float(clicks.sum())

    def _build_candidate_ids(
        self, row: pd.Series, positive_id: int, user_id: int
    ) -> np.ndarray:
        return np.array(row["slate"], dtype=np.int64)
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_env_finn_no_slate.py -v
```
Expected: 5 passed

- [ ] **Step 5: Run full suite**

```bash
pytest -v --tb=short
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/environments/finn_no_slate.py tests/test_env_finn_no_slate.py
git commit -m "feat: add FinnNoSlateEnv"
```

---

### Task 7: Update RL4RS pipeline to emit sessions.parquet

**Files:**
- Modify: `rl_recsys/data/pipelines/rl4rs.py`
- Create: `tests/test_pipeline_rl4rs.py`

The pipeline auto-detects column names by prefix pattern so it works with any number of features. The expected CSV format uses these column naming conventions:
- Session column: `session_id`
- User state: `user_feat_0`, `user_feat_1`, … (any count)
- Item IDs: `item_id_0`, `item_id_1`, … (any count, determines slate size)
- Item features: `item_0_feat_0`, `item_0_feat_1`, … (any count, one group per item)
- Click labels: `click_0`, `click_1`, … (one per item)

If the real `rl4rs_dataset_a_rl.csv` uses different column names, update the column detection regexes in `_detect_columns()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_rl4rs.py
from __future__ import annotations

import pandas as pd
import pytest

from rl_recsys.data.pipelines.rl4rs import RL4RSPipeline


def _write_rl_csv(path, n_sessions=2, n_steps=3):
    """Write a minimal rl4rs_dataset_a_rl.csv with 2 user feats, 3 items, 2 item feats."""
    import io, random
    random.seed(0)
    rows = []
    for sid in range(n_sessions):
        for _ in range(n_steps):
            row = {"session_id": sid}
            row["user_feat_0"] = round(random.random(), 4)
            row["user_feat_1"] = round(random.random(), 4)
            for k in range(3):
                row[f"item_id_{k}"] = random.randint(100, 999)
                row[f"item_{k}_feat_0"] = round(random.random(), 4)
                row[f"item_{k}_feat_1"] = round(random.random(), 4)
                row[f"click_{k}"] = random.randint(0, 1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_process_produces_sessions_parquet(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    p = RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "sessions.parquet"
    assert out.exists()


def test_sessions_parquet_schema(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()

    df = pd.read_parquet(proc_dir / "sessions.parquet")
    assert set(df.columns) >= {"session_id", "step", "user_state", "slate", "item_features", "clicks"}


def test_sessions_parquet_step_indices(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()

    df = pd.read_parquet(proc_dir / "sessions.parquet")
    for sid, grp in df.groupby("session_id"):
        assert sorted(grp["step"].tolist()) == list(range(len(grp)))


def test_sessions_parquet_list_shapes(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()

    df = pd.read_parquet(proc_dir / "sessions.parquet")
    row = df.iloc[0]
    assert len(row["user_state"]) == 2    # 2 user features in mock
    assert len(row["slate"]) == 3         # 3 items in mock
    assert len(row["item_features"]) == 3  # 3 items
    assert len(row["item_features"][0]) == 2  # 2 features per item
    assert len(row["clicks"]) == 3        # 3 click labels


def test_rl4rs_is_registered():
    import rl_recsys.data.pipelines.rl4rs  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "rl4rs" in _REGISTRY
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_pipeline_rl4rs.py -v
```
Expected: `AssertionError` on `test_process_produces_sessions_parquet` (file doesn't exist)

- [ ] **Step 3: Replace `rl_recsys/data/pipelines/rl4rs.py` with working implementation**

```python
from __future__ import annotations

import re
import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from rl_recsys.data.pipelines.base import BasePipeline


class RL4RSPipeline(BasePipeline):
    """Pipeline for downloading and processing the RL4RS dataset.

    Emits sessions.parquet with columns:
        session_id (int64), step (int64),
        user_state (list[float]), slate (list[int]),
        item_features (list[list[float]]), clicks (list[int])

    Column auto-detection expects the rl4rs_dataset_a_rl.csv to follow:
        user_feat_N   — user state features
        item_id_N     — item ID per slate slot
        item_N_feat_M — item features (N=slot, M=feature index)
        click_N       — binary click per slot
    If the real CSV uses different names, update _detect_columns().
    """

    DATASET_URL = "https://zenodo.org/record/6622390/files/rl4rs-dataset.tar.gz"

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/rl4rs",
        processed_dir: str | Path = "data/processed/rl4rs",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive_path = self.raw_dir / "rl4rs-dataset.tar.gz"
        if not archive_path.exists():
            print(f"Downloading RL4RS dataset from {self.DATASET_URL}...")
            response = requests.get(self.DATASET_URL, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            with open(archive_path, "wb") as f, tqdm(
                total=total_size, unit="iB", unit_scale=True
            ) as pbar:
                for data in response.iter_content(1024):
                    size = f.write(data)
                    pbar.update(size)
        else:
            print(f"Archive found at {archive_path}, skipping download.")
        print(f"Extracting to {self.raw_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.raw_dir)
        print("Done.")

    def process(self) -> None:
        rl_file = self.raw_dir / "rl4rs-dataset" / "rl4rs_dataset_a_rl.csv"
        if not rl_file.exists():
            raise FileNotFoundError(
                f"Not found: {rl_file}. Run --download first."
            )
        df = pd.read_csv(rl_file)
        cols = _detect_columns(df)

        df["step"] = df.groupby(cols["session_id"]).cumcount()
        df["user_state"] = df[cols["user_feat"]].values.tolist()
        df["slate"] = df[cols["item_id"]].values.tolist()
        n_items = len(cols["item_id"])
        df["item_features"] = [
            [row[cols["item_feat"][i]].tolist() for i in range(n_items)]
            for _, row in df.iterrows()
        ]
        df["clicks"] = df[cols["click"]].values.tolist()

        out_df = df[[cols["session_id"], "step", "user_state", "slate", "item_features", "clicks"]].rename(
            columns={cols["session_id"]: "session_id"}
        )
        out = self.processed_dir / "sessions.parquet"
        out_df.to_parquet(out, index=False)
        print(f"Saved {len(out_df):,} rows ({out_df['session_id'].nunique():,} sessions) to {out}")


def _detect_columns(df: pd.DataFrame) -> dict:
    user_feat = sorted(
        [c for c in df.columns if re.match(r"^user_feat_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    item_id = sorted(
        [c for c in df.columns if re.match(r"^item_id_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    n_items = len(item_id)
    item_feat = [
        sorted(
            [c for c in df.columns if re.match(rf"^item_{i}_feat_\d+$", c)],
            key=lambda x: int(x.split("_")[-1]),
        )
        for i in range(n_items)
    ]
    click = sorted(
        [c for c in df.columns if re.match(r"^click_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    session_id = "session_id"
    if session_id not in df.columns:
        raise ValueError(
            f"'session_id' column not found in CSV. Available columns: {list(df.columns)}"
        )
    if not user_feat:
        raise ValueError("No user_feat_N columns found in CSV.")
    if not item_id:
        raise ValueError("No item_id_N columns found in CSV.")
    if not click:
        raise ValueError("No click_N columns found in CSV.")
    return {
        "session_id": session_id,
        "user_feat": user_feat,
        "item_id": item_id,
        "item_feat": item_feat,
        "click": click,
    }


from rl_recsys.data.registry import register  # noqa: E402

register(
    "rl4rs",
    RL4RSPipeline,
    schema="slates",
    tags=["RL/Slate"],
    raw_dir="data/raw/rl4rs",
    processed_dir="data/processed/rl4rs",
)
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_pipeline_rl4rs.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/pipelines/rl4rs.py tests/test_pipeline_rl4rs.py
git commit -m "feat: rewrite RL4RS pipeline to emit sessions.parquet"
```

---

### Task 8: RL4RSEnv

**Files:**
- Create: `rl_recsys/environments/rl4rs.py`
- Create: `tests/test_env_rl4rs.py`
- Modify: `rl_recsys/environments/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_env_rl4rs.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.rl4rs import RL4RSEnv


def _sessions_parquet(tmp_path, n_sessions=2, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4):
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(n_sessions):
        for step in range(n_steps):
            slate = rng.integers(0, 100, size=slate_size).tolist()
            rows.append({
                "session_id": sid,
                "step": step,
                "user_state": rng.standard_normal(n_user_feats).tolist(),
                "slate": [int(x) for x in slate],
                "item_features": rng.standard_normal((slate_size, n_item_feats)).tolist(),
                "clicks": rng.integers(0, 2, size=slate_size).tolist(),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_path / "sessions.parquet", index=False)


def test_reset_obs_shape(tmp_path):
    _sessions_parquet(tmp_path, n_user_feats=4, n_item_feats=4, slate_size=3)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (4,)
    assert obs.candidate_features.shape == (3, 4)
    assert obs.candidate_ids.shape == (3,)


def test_done_false_mid_session(tmp_path):
    _sessions_parquet(tmp_path, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1, 2]))
    assert step.done is False


def test_done_true_at_session_end(tmp_path):
    _sessions_parquet(tmp_path, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    env.reset(seed=0)
    env.step(np.array([0, 1, 2]))
    env.step(np.array([0, 1, 2]))
    step = env.step(np.array([0, 1, 2]))
    assert step.done is True


def test_native_feature_dim_mismatch_raises(tmp_path):
    _sessions_parquet(tmp_path, n_user_feats=4, n_item_feats=4, slate_size=3)
    with pytest.raises(ValueError, match="feature_dim"):
        RL4RSEnv(tmp_path, slate_size=3, feature_dim=8, feature_source="native", seed=0)


def test_hashed_mode_works(tmp_path):
    _sessions_parquet(tmp_path, n_user_feats=4, n_item_feats=4, slate_size=3)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=8, feature_source="hashed", seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (3, 8)


def test_reward_accumulates_across_steps(tmp_path):
    _sessions_parquet(tmp_path, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    env.reset(seed=0)
    total = 0.0
    for _ in range(3):
        step = env.step(np.array([0, 1, 2]))
        total += step.reward
    assert total >= 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_env_rl4rs.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.environments.rl4rs'`

- [ ] **Step 3: Create `rl_recsys/environments/rl4rs.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.environments.dataset_base import SessionDatasetEnv


class RL4RSEnv(SessionDatasetEnv):
    """Multi-step session environment backed by RL4RS logged sessions.

    Each episode is one full session from sessions.parquet (produced by the
    updated RL4RS pipeline). Candidate pool = the logged slate at each step.
    Reward = number of agent-selected items that were clicked in the log.

    feature_source="native": uses pre-computed user_state and item_features
    vectors directly; feature_dim must exactly match the vector length in data.
    feature_source="hashed": ignores pre-computed vectors, uses hashed IDs.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        slate_size: int = 6,
        feature_dim: int = 32,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        processed_dir = Path(processed_dir)
        df = pd.read_parquet(processed_dir / "sessions.parquet")

        if feature_source == "native":
            # Validate feature_dim against the actual vector lengths in the data
            sample_row = df.iloc[0]
            actual_user_dim = len(sample_row["user_state"])
            actual_item_dim = len(sample_row["item_features"][0])
            if feature_dim != actual_user_dim:
                raise ValueError(
                    f"feature_dim={feature_dim} does not match user_state length={actual_user_dim} "
                    "in sessions.parquet. Set feature_dim to match the data."
                )
            if feature_dim != actual_item_dim:
                raise ValueError(
                    f"feature_dim={feature_dim} does not match item_features width={actual_item_dim} "
                    "in sessions.parquet. Set feature_dim to match the data."
                )

        sessions: dict[int, pd.DataFrame] = {
            int(sid): grp.sort_values("step").reset_index(drop=True)
            for sid, grp in df.groupby("session_id")
        }
        num_candidates = len(df.iloc[0]["slate"])

        super().__init__(
            sessions,
            slate_size=slate_size,
            num_candidates=num_candidates,
            feature_dim=feature_dim,
            feature_source=feature_source,
            seed=seed,
        )

    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float:
        return float(clicks.sum())

    def _get_user_features(self, row: pd.Series) -> np.ndarray:
        if self._feature_source == "native":
            return np.array(row["user_state"], dtype=np.float32)
        return super()._get_user_features(row)

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        if self._feature_source == "native":
            return np.array(row["item_features"], dtype=np.float32)
        return super()._get_item_features(row, candidate_ids)
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_env_rl4rs.py -v
```
Expected: 6 passed

- [ ] **Step 5: Export new classes from `rl_recsys/environments/__init__.py`**

Add the following imports and `__all__` entries:

```python
from rl_recsys.environments.finn_no_slate import FinnNoSlateEnv
from rl_recsys.environments.kuairec import KuaiRecEnv
from rl_recsys.environments.rl4rs import RL4RSEnv
```

And add `"FinnNoSlateEnv"`, `"KuaiRecEnv"`, `"RL4RSEnv"` to `__all__`.

- [ ] **Step 6: Run full test suite**

```bash
pytest -v --tb=short
```
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/environments/rl4rs.py rl_recsys/environments/__init__.py tests/test_env_rl4rs.py
git commit -m "feat: add RL4RSEnv"
```
