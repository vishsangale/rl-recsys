# Dataset Suite Design

**Date:** 2026-04-30
**Status:** Draft

## Goal

Establish an extensive suite of open-source, freely downloadable recommendation system datasets in rl-recsys. Datasets serve two purposes: (1) RL training — real interaction logs to build data-driven environments replacing or augmenting the synthetic one, and (2) benchmarking — standard offline evaluation splits for comparing agents on NDCG/MRR/CTR metrics.

## Constraints

- No registration or data access agreement required (registration-gated datasets such as Netflix Prize, MIND, Yelp are deferred to a future phase)
- One pipeline per dataset, following the existing `BasePipeline` interface (`download()` + `process()` → Parquet)
- Datasets are added incrementally as needed; this document is the master catalog

## Existing Infrastructure

- `rl_recsys/data/pipelines/base.py` — `BasePipeline(raw_dir, processed_dir)` with abstract `download()` and `process()`
- Three pipelines already implemented: MovieLens-100K, Last.fm-1K, RL4RS
- `scripts/prepare_data.py` — CLI entry point (`--dataset`, `--download`, `--process`)

## Dataset Catalog

Tags: `[CF]` = collaborative filtering/ratings · `[Session]` = sequential/session logs · `[RL/Slate]` = slate/ranking logs for RL

### Movies & TV

| Dataset | Tag | Size | Source |
|---|---|---|---|
| MovieLens-100K | [CF] | 100K ratings, 943 users | GroupLens — **implemented** |
| MovieLens-1M | [CF] | 1M ratings, 6K users | GroupLens |
| MovieLens-10M | [CF] | 10M ratings, 72K users | GroupLens |
| MovieLens-20M | [CF] | 20M ratings, 138K users | GroupLens |
| MovieLens-25M | [CF] | 25M ratings, 162K users + tag genome | GroupLens |
| HetRec-2011 MovieLens | [CF] | 855K ratings + social links | HetRec workshop |

### Music

| Dataset | Tag | Size | Source |
|---|---|---|---|
| Last.fm-1K | [Session] | 19M plays, 1K users | MTG Barcelona — **implemented** |
| Last.fm-360K | [CF] | 17B plays, 360K users | MTG Barcelona |
| HetRec-2011 Last.fm | [CF+Social] | 186K artist listens + friends graph | HetRec workshop |

### Books

| Dataset | Tag | Size | Source |
|---|---|---|---|
| Book-Crossing | [CF] | 1.1M ratings, 278K users, 271K books | Cai-Nicolas Ziegler |
| Goodreads | [CF] | 229M interactions, 876K users | UCSD McAuley lab |

### E-commerce

| Dataset | Tag | Size | Source |
|---|---|---|---|
| Amazon Reviews (per category) | [CF/Session] | Varies; 28 categories | UCSD McAuley lab |
| Steam | [CF] | 7.8M reviews, 2.5M users | UCSD McAuley lab |
| Diginetica | [Session] | 1.2M sessions, 43K items | CIKM Cup 2016 |

### Location & Social

| Dataset | Tag | Size | Source |
|---|---|---|---|
| Gowalla | [Session] | 6.4M check-ins, 196K users | SNAP Stanford |
| BrightKite | [Session] | 4.5M check-ins, 58K users | SNAP Stanford |

### Entertainment & Misc

| Dataset | Tag | Size | Source |
|---|---|---|---|
| Jester | [CF] | 4.1M ratings, 73K users, 100 jokes | UC Berkeley |
| CiaoDVD | [CF+Social] | 72K ratings, 17K users | FilmTrust/EPFL |
| FilmTrust | [CF+Social] | 35K ratings, 1.5K users | FilmTrust |
| Epinions | [CF+Social] | 188K ratings, 49K users | Stanford |
| Anime | [CF] | 7.8M ratings, 73K users, 12K items | GitHub |

### News

| Dataset | Tag | Size | Source |
|---|---|---|---|
| Adressa (1-week) | [Session] | 2.7M events, 561K users | Norwegian University of Science and Technology |

### RL / Slate-specific

| Dataset | Tag | Size | Source |
|---|---|---|---|
| RL4RS | [RL/Slate] | Slate logs from real app | Zenodo — **implemented** |
| KuaiRec | [RL/Session] | 12M interactions, fully observed matrix | Kuaishou / GitHub |
| KuaiSAR | [RL/Session] | Combined search + rec logs | Kuaishou / GitHub |
| FINN.no Slate | [RL/Slate] | 37M slates, Norwegian classifieds | Zenodo |

**Total: 25 datasets across 7 domains** (3 implemented, 22 pending)

## Integration Pattern

Each new dataset gets:
1. `rl_recsys/data/pipelines/<name>.py` — a `<Name>Pipeline(BasePipeline)` class
2. An entry in `scripts/prepare_data.py` choices and dispatch
3. A test fixture in `tests/` verifying schema of processed Parquet output

MovieLens variants (1M, 10M, 20M, 25M) share one pipeline class parameterized by variant string, not four separate classes.

Amazon categories share one pipeline class parameterized by category slug.

## Deferred (Registration-gated)

Netflix Prize, MIND (Microsoft News), Yelp, Yahoo! Music — to be added in a future phase once access decisions are made.
