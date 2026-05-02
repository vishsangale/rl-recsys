# Dataset Catalog

This repo uses registry-driven dataset pipelines. Importing pipeline modules
self-registers dataset keys, and `scripts/prepare_data.py` dispatches by key.
Commands below assume the repo virtualenv is active:

```bash
source .venv/bin/activate
```

## Prepare Data

```bash
# Show available dataset keys
python scripts/prepare_data.py --help

# Download and process a dataset using its default directories
python scripts/prepare_data.py --dataset movielens-100k --download --process

# Process files that were already downloaded
python scripts/prepare_data.py --dataset gowalla --process

# Override raw and processed directories
python scripts/prepare_data.py \
  --dataset finn-no-slate \
  --raw-dir /data/raw/finn_no_slate \
  --processed-dir /data/processed/finn_no_slate \
  --download --process
```

Processed outputs are Parquet files under each dataset's processed directory.
The common schema labels are:

| Schema | Required columns | Typical use |
| --- | --- | --- |
| `interactions` | `user_id`, `item_id`, `timestamp` | Explicit ratings, implicit feedback, logged bandit clicks |
| `sessions` | `session_id`, `user_id`, `item_id`, `timestamp` | Ordered events or per-event session-like records |
| `slates` | `request_id`, `user_id` | Slate recommendation logs with candidate lists or native slate files |

## Which Dataset To Use

| Goal | Start with | Move to | Notes |
| --- | --- | --- | --- |
| Fast CF smoke tests | `movielens-100k` | `movielens-1m` | Small, standard explicit-rating data with clean IDs. |
| Larger explicit-rating CF | `movielens-10m` | `movielens-20m`, `movielens-25m` | Same domain and schema as MovieLens-100K, larger scale. |
| Cross-domain CF | `book-crossing`, `amazon-books` | `amazon-electronics`, `amazon-movies`, `amazon-video-games` | Useful when checking whether methods generalize beyond movies. |
| Implicit feedback | `steam` | `kuairec` | Steam uses hours played; KuaiRec uses watch ratio. |
| Sequential or event-style recommendation | `gowalla` | `lastfm-1k` | Gowalla is check-ins; Last.fm is listening events. |
| Slate recommendation | `finn-no-slate` | `rl4rs` | FINN.no has explicit slate/click structure; RL4RS keeps native slate/eval files. |
| Off-policy evaluation | `open-bandit` | filtered OPE splits | Preserves `propensity_score`, `policy`, `campaign`, and native context columns; anonymous users are encoded as `user_id = 0`. |
| RL environment construction | `kuairec` | `finn-no-slate` | KuaiRec is useful for dense user-item feedback; FINN.no is useful for real slate logs. |

## Available Dataset Keys

| Key | Schema | Scale and signal | Best use | Output notes |
| --- | --- | --- | --- | --- |
| `movielens-100k` | `interactions` | 100K movie ratings | Fast CF development and test runs | `ratings_100k.parquet` with rating and timestamp |
| `movielens-1m` | `interactions` | 1M movie ratings | Medium CF experiments | `ratings_1m.parquet` |
| `movielens-10m` | `interactions` | 10M movie ratings | Larger CF experiments | `ratings_10m.parquet` |
| `movielens-20m` | `interactions` | 20M movie ratings | Large movie-domain baselines | `ratings_20m.parquet` |
| `movielens-25m` | `interactions` | 25M movie ratings | Largest current MovieLens baseline | `ratings_25m.parquet` |
| `book-crossing` | `interactions` | 1.1M book ratings, no source timestamp | Book-domain CF | `ratings.parquet`; timestamp is filled with `0` |
| `amazon-books` | `interactions` | Amazon Reviews 2018 Books 5-core ratings | Product-review CF in books | `interactions.parquet`; explicit `overall` rating |
| `amazon-electronics` | `interactions` | Amazon Reviews 2018 Electronics 5-core ratings | Product-review CF in electronics | Same Amazon pipeline with category override |
| `amazon-movies` | `interactions` | Amazon Reviews 2018 Movies and TV 5-core ratings | Product-review CF in media | Same Amazon pipeline with category override |
| `amazon-video-games` | `interactions` | Amazon Reviews 2018 Video Games 5-core ratings | Product-review CF in games | Same Amazon pipeline with category override |
| `steam` | `interactions` | 7.8M Steam reviews, hours played | Implicit-feedback CF | Parses Python-literal rows with `ast.literal_eval`; rating is `hours` |
| `kuairec` | `interactions` | 12M Kuaishou interactions, watch ratio | RL/simulator work and implicit video feedback | `interactions.parquet`; download uses `verify=False` for the source host |
| `open-bandit` | `interactions` | 26M logged bandit events across random/BTS policies and all/men/women campaigns | Off-policy evaluation | `interactions.parquet`; click is rating, `propensity_score`, `policy`, `campaign`, `position`, `user_feature_*`, `item_feature_*`, and `user_item_affinity_*` are retained |
| `gowalla` | `sessions` | 6.4M check-ins, 196K users | Location/event recommendation | `sessions.parquet`; location IDs are factorized as items |
| `lastfm-1k` | `sessions` | 19M listening events, 1K users | Music listening sequences | Legacy output currently keeps native listening columns plus factorized IDs |
| `finn-no-slate` | `slates` | 37M slate impressions, 25 candidates per slate | Slate ranking and click modeling | `slates.parquet`; includes slate candidate list, click index, timestamp |
| `rl4rs` | `slates` | Real-app slate logs | RL slate benchmark workflows | Writes native `slate_train.parquet` and `slate_eval.parquet` files |

## Practical Notes

- Start with `movielens-100k` when validating a new model or data loader.
- Prefer `open-bandit` only when the method uses logged propensities or OPE
  assumptions; it has no persistent user identity, so contextual evaluation
  should use the native feature source rather than only grouped user IDs.
- `experiments/run_ope_benchmark.py` defaults to the historical
  `policy=random`, `campaign=all` slice. It now uses native Open Bandit context
  columns by default; add `--feature-source hashed` to reproduce the older
  grouped/hash features. Use `--policy any --campaign any` when you want all
  processed Open Bandit splits.
- Use `finn-no-slate` for real slate choice data. Use `kuairec` when you need
  denser user-item feedback for environment construction.
- Several large pipelines currently load source files with pandas in memory.
  For memory-constrained machines, process smaller variants first or add
  chunked processing before running the largest datasets.
- Amazon has one parameterized pipeline class. Four categories are registered
  as CLI keys, and other categories can be instantiated directly in Python with
  `AmazonPipeline(category="...")`.

## Adding A Dataset

New pipelines should subclass `BasePipeline`, implement `download()` and
`process()`, write standardized Parquet, call `validate_parquet_schema()` when
possible, and register themselves at module import with `register(...)`.
Also add one import line to `scripts/prepare_data.py` so the CLI sees the key.
