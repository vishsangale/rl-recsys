# Dataset Suite Design

**Date:** 2026-04-30
**Status:** Draft

## Goal

Establish an extensive suite of open-source, freely downloadable recommendation system datasets in rl-recsys. Datasets serve two purposes: (1) RL training — real interaction logs to build data-driven environments replacing or augmenting the synthetic one, and (2) benchmarking — standard offline evaluation splits for comparing agents on NDCG/MRR/CTR metrics.

## Constraints

- No registration or data access agreement required (registration-gated datasets deferred to a future phase)
- Datasets are added incrementally as needed; this document is the master catalog
- Each dataset gets one pipeline class following `BasePipeline`, added to a central registry

## Existing Infrastructure

- `rl_recsys/data/pipelines/base.py` — `BasePipeline(raw_dir, processed_dir)` with abstract `download()` and `process()`
- Three pipelines already implemented: MovieLens-100K, Last.fm-1K, RL4RS
- `scripts/prepare_data.py` — CLI entry point (currently hard-coded; must become registry-driven before scaling)

## Architectural Notes (from review)

**Registry required:** `prepare_data.py` currently hard-codes imports, choices, and dispatch. Before adding more pipelines it must be refactored to a registry/config-driven loader that accepts `--dataset`, `--variant`, `--category`, `--raw-dir`, `--processed-dir`, and `--force`.

**Chunked processing required:** Current pipelines load full files into pandas. Datasets at Goodreads (229M rows), Amazon, Last.fm-360K, and FINN scale will not fit in memory. All new pipelines must use chunked or streaming processing (PyArrow batch reader or pandas `chunksize`), writing partitioned Parquet.

**Download robustness:** All pipelines should support resumable downloads, checksum verification, and safe archive extraction before scaling.

**Standardized schema:** Processed Parquet must emit standard column names. Schema type determines the columns required:

| Schema type | Required columns |
|---|---|
| `interactions` | `user_id`, `item_id`, `rating` (or implicit 1.0), `timestamp` |
| `sessions` | `session_id`, `user_id`, `item_id`, `timestamp`, `event_type` |
| `slates` | `request_id`, `user_id`, `slate` (list), `clicks`, `timestamp` |
| `social_edges` | `user_a`, `user_b`, `trust_score` |
| `items` | `item_id`, + domain-specific features |

## Dataset Catalog

Tags: `[CF]` = collaborative filtering/ratings · `[Session]` = sequential/session logs · `[RL/Slate]` = slate/ranking logs for RL · `[OPE]` = logged bandit data for off-policy evaluation

Note on counting: MovieLens and Amazon are **pipeline families** (one parameterized class, multiple downloadable instances). The catalog lists all instances for completeness.

### Movies & TV

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| MovieLens-100K | [CF] | 100K ratings, 943 users, 1,682 items | GroupLens — **implemented** |
| MovieLens-1M | [CF] | 1M ratings, 6,040 users | GroupLens |
| MovieLens-10M | [CF] | 10M ratings, 72K users | GroupLens |
| MovieLens-20M | [CF] | 20M ratings, 138K users | GroupLens; includes tag genome |
| MovieLens-25M | [CF] | 25M ratings, 162K users | GroupLens; largest, includes tag genome scores |
| HetRec-2011 MovieLens | [CF] | 855K ratings + social links + IMDB/RT metadata | HetRec workshop |

### Music

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| Last.fm-1K | [Session] | 19M listening events, 1K users | MTG Barcelona — **implemented** |
| Last.fm-360K | [CF] | 17.56M user-artist-playcount rows, 360K users | MTG Barcelona; aggregate play counts, not individual events |
| HetRec-2011 Last.fm | [CF+Social] | 92,834 user-artist listen relations + friends graph; ~186K tag assignments | HetRec workshop |

### Books

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| Book-Crossing | [CF] | 1.1M ratings, 278K users, 271K books | Cai-Nicolas Ziegler |
| Goodreads | [CF] | 228M interactions (shelves, reads, ratings), 876K users | UCSD McAuley lab; "interactions" includes non-rating events, not pure ratings |

### E-commerce

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| Amazon Reviews 2018 | [CF/Session] | 29 categories, varies per category | UCSD McAuley lab; one pipeline class parameterized by category slug |
| Steam | [CF] | 7.8M reviews, 2.5M users | UCSD McAuley lab |
| Diginetica | [Session] | 1.2M sessions, 43K items | CIKM Cup 2016; query-product click logs |
| YOOCHOOSE | [Session] | 9.2M sessions (clicks + purchases) | RecSys 2015 challenge; verify direct download without Kaggle login |

### Location & Social

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| Gowalla | [Session] | 6.4M check-ins, 196K users | SNAP Stanford |
| BrightKite | [Session] | 4.5M check-ins, 58K users | SNAP Stanford |

### Entertainment & Misc

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| Jester | [CF] | 4.1M continuous ratings, 73K users, 100 jokes | UC Berkeley |
| CiaoDVD | [CF+Social] | 72K ratings, 17K users | Includes trust graph |
| FilmTrust | [CF+Social] | 35K ratings, 1.5K users | Small; useful for social-aware agents |
| Epinions | [CF+Social] | ~181K feedback rows, 116K users (McAuley social-rec variant) | Note: SNAP has trust graph only; use McAuley's social-rec release for ratings |
| Anime | [CF] | 7.8M ratings, 73K users, 12K items | GitHub; verify license |
| HetRec-2011 Delicious | [Session] | 437K bookmarks, 1.9K users | HetRec workshop; tag-aware |

### News

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| Adressa (1-week) | [Session] | 2.7M events, 561K users | Norwegian University of Science and Technology |

### RL / Slate / OPE

| Dataset | Tag | Size | Notes |
|---|---|---|---|
| RL4RS | [RL/Slate] | Slate logs from a real app | Zenodo — **implemented** |
| KuaiRec | [RL/Session] | 12M interactions; fully observed user-item matrix | Kuaishou / GitHub; no missing-data bias, ideal for RL env |
| KuaiSAR | [RL/Session] | Combined search + rec logs | Kuaishou / GitHub; useful for unified modeling |
| FINN.no Slate | [RL/Slate] | 37M slates, Norwegian classifieds | Zenodo; real slate recommendation logs |
| Open Bandit Dataset | [OPE] | 26M logged bandit feedback (3 policies) | Zozotown / GitHub; designed for off-policy evaluation |
| Criteo CTR | [OPE] | 45M labeled ad impressions | Criteo AI Lab; direct download; useful for CTR reward modeling |

**Pipeline families:** 2 (MovieLens variants, Amazon categories)
**Pipeline instances (downloadable datasets):** 28

## Integration Pattern

Each new dataset:
1. `rl_recsys/data/pipelines/<name>.py` — `<Name>Pipeline(BasePipeline)` class with chunked processing
2. Entry in the central dataset registry (to be implemented)
3. `scripts/prepare_data.py` reads registry; no more hard-coded dispatch
4. Test fixture in `tests/` verifying schema, row counts, and ID mapping of processed Parquet

## Deferred (Registration-gated)

Netflix Prize, MIND (Microsoft News), Yelp, Yahoo! Music — to be added in a future phase once access decisions are made.

## Open Questions

- YOOCHOOSE: confirm direct download path without Kaggle account
- Criteo CTR: confirm no ToS restriction on redistribution for research use
- Epinions: pin to McAuley social-rec release URL specifically
