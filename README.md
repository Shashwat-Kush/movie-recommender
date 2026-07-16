# movie-recommender

A MovieLens 25M recommendation engine built to run on an 8GB M2 Mac: streaming data
pipeline → Two-Tower retrieval (PyTorch/MPS) → custom C++ HNSW index → Cerebras LLM reranking,
served over FastAPI.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## The model

The served retrieval model is a **history-based Two-Tower** trained with **in-batch sampled
softmax** (InfoNCE, temperature 0.05) and **logQ correction**:

- **User tower**: a recency-weighted mean (decay 0.9 per position) of the item-embedding
  rows for the user's 20 most recent liked movies (rating ≥ 3.5), through a 3-layer MLP.
  No per-user parameters — the model serves any user with a watch history and reacts to
  new watches without retraining.
- **Item tower**: item-ID embedding concatenated with a 128-dim MiniLM metadata embedding
  (title/genres/tags), through its own MLP. During training the ID embedding is zeroed
  for a random 20% of items (**ID dropout**), which is what makes the metadata-only
  cold-start path actually work. Both outputs are L2-normalized; the score is their dot
  product.
- **Loss**: every other in-batch item acts as a popularity-distributed negative. Cosine
  logits are divided by a temperature (bounded logits fed straight into a loss saturate and
  stop learning — the original BCE-on-cosine setup failed exactly this way). Subtracting
  each item's log sampling probability (logQ) debiases the softmax so raw cosine ranks by
  likelihood — no serving-time popularity correction needed.
- **Protocol**: trained on `train_loo` (each user's history minus the last two interactions);
  the training positive is excluded from its own user's history to prevent label leakage;
  only ratings ≥ 3.5 count as positives.

## Results

Leave-one-out over all 138,493 users, movies the user already rated masked before top-K,
K = 10 (`outputs/eval_loo*.json`):

| Model | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity baseline | 4.86% | 2.48% |
| Implicit MF / BPR (3 epochs) | 6.45% | 3.31% |
| ID-embedding Two-Tower, raw cosine | 4.51% | 2.16% |
| ID-embedding Two-Tower + tuned popularity blend (w=0.1) | 9.57% | 4.77% |
| History Two-Tower + logQ, raw cosine | 9.53% | 4.70% |
| **+ ID dropout & recency decay (served, "v2")** | **9.50%** | **4.72%** |

The last three are statistically tied on warm recall, but each iteration is structurally
better: the history+logQ model dropped per-user parameters (85MB) and the tuned serving
knob; v2 additionally fixes cold-start (below) at no warm cost.

**Cold-start** (`scripts/evaluate_cold_start.py`, 20% of items embedded metadata-only) —
before ID dropout the cold path collapsed to **0.13%** recall@10 vs 10.56% warm on the
same users (the item tower had never seen a zeroed ID embedding). With ID dropout, cold
items score **8.89%** vs 10.34% warm — within 1.5pt of warm and above the 7.21%
popularity baseline for those users. A genuinely new movie now gets sensible
recommendations from metadata alone.

**Temporal robustness** (`scripts/evaluate_timesplit.py`) — recall by the period of the
held-out interaction: 9.72% before 2009, 9.40% in 2009–2012, 8.50% in 2012–2016; ~2× the
popularity baseline in every window, with the recency-decay pooling improving the two
recent windows over the plain mean (9.10% → 9.40%, 8.28% → 8.50%). (The LOO protocol
trains on all periods, so this measures robustness to recency, not strict train-on-past
generalization.)

**Reranker A/B** (`scripts/evaluate_reranker.py`, 200 users) — LLM reranking (now Cerebras gpt-oss-120b; the A/B ran on Groq llama-3.1-8b) of the
retrieval top candidates moves recall@10 from 7.0% to 7.5% and NDCG from 3.88 to 4.22:
a small lift, within noise at this sample size.

## How it fits together

```
raw CSVs ──▶ partitioned Parquet ──▶ train/val/test splits
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            cold-start embeddings                      Two-Tower training
            (MiniLM → 128-dim)                         (user tower / item tower)
                     └────────────────────┬────────────────────┘
                                          ▼
                            item embeddings (128-dim)
                                          ▼
                              C++ HNSW index (top-K)
                                          ▼
                            Cerebras LLM reranker (top-N)
                                          ▼
                                  FastAPI /recommend
```

## Setup

```bash
pip install -r requirements.txt

# Cerebras API key for the reranker
echo 'CEREBRAS_API_KEY=your_key_here' > .env

# Build the C++ retrieval engine
cd cpp && ./build.sh && cd ..
```

## Running the pipeline

Scripts import from `src/`, so run them with the project root on `PYTHONPATH`:

```bash
# 1. Raw CSVs → Parquet  (only needed once)
PYTHONPATH=. python3 -m src.data.csv_to_parquet
PYTHONPATH=. python3 -m src.data.splits

# 2. Cold-start metadata embeddings  (only needed once)
PYTHONPATH=. python3 scripts/generate_cold_start.py

# 3. Train  (writes checkpoints/two_tower_history/best_model.pt; --resume continues
#    from the latest epoch checkpoint after an interruption)
PYTHONPATH=. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 -u scripts/train_all.py --model two_tower_history

# 4. Build the HNSW index from the trained model
PYTHONPATH=. python3 scripts/build_index.py --checkpoint checkpoints/two_tower_history/best_model.pt

# 5. Evaluate  (writes outputs/eval_loo.json + the serving artifacts
#    popularity_counts.npy and seen_items.npz)
PYTHONPATH=. python3 scripts/evaluate.py --checkpoint checkpoints/two_tower_history/best_model.pt
PYTHONPATH=. python3 scripts/evaluate_timesplit.py     # recall by time period
PYTHONPATH=. python3 scripts/evaluate_cold_start.py    # pseudo-cold item protocol
PYTHONPATH=. python3 scripts/evaluate_reranker.py      # retrieval vs LLM-reranked A/B
PYTHONPATH=. python3 scripts/evaluate_mf.py            # BPR baseline

# 6. Serve
python3 -m uvicorn app.main:app --port 8000
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "query": "funny action movies", "top_k": 5}'

# Cold-start: recommendations for a brand-new user from a few picked movies
curl -X POST http://localhost:8000/recommend_cold \
  -H "Content-Type: application/json" \
  -d '{"liked_movie_ids": [2571, 4226, 48780], "query": "a smart thriller", "top_k": 5}'

# 7. Frontend (frontend/): a static Next.js portfolio demo that replays real
#    recorded responses — deployable to Vercel with zero backend.
PYTHONPATH=. python3 scripts/capture_fixtures.py   # record fixtures from the live stack
cd frontend && npm install && npm run build        # then `npm run dev` or deploy
```

`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` matters on 8GB machines: training fits comfortably,
but Metal's default working-set cap counts *other apps'* memory, and the ~70MB embedding-
gradient allocation gets refused when the system is busy.

Use `python3 -u` for training — stdout is block-buffered when redirected to a log, and at
~60 bytes per progress line you won't see output for several epochs otherwise.

## Item indexing: the thing to know

There are **two different orderings** of movies in this project, and conflating them is the
single easiest way to break it:

| Ordering | Source | Size | Used by |
|---|---|---|---|
| **movie_map index** | `movie_mapping.parquet` — movies present in ratings, sorted by movieId | 26,744 | model item IDs, HNSW index positions |
| **cold-start row** | `cold_start_movie_ids.npy` — every movie in `movie.csv` | 27,278 | saved metadata embeddings |

They diverge (e.g. movieId 118734 is movie_map index 24836 but cold-start row 25141), so
**positional truncation between them is always wrong**. Everything that touches metadata goes
through `build_aligned_metadata()` in `src/data/cold_start.py`, which reindexes into
movie_map order. HNSW index positions are movie_map indices, so results map back to movieIds
via `movie_mapping.parquet` — never via `cold_start_corpus.parquet`.

Item embeddings use the **warm** path (`get_item_embeddings`: item ID + metadata) in both
`build_index.py` and `evaluate.py`, so evaluation measures what the API actually serves.

## Layout

```
data/raw/          Original MovieLens CSVs (gitignored)
data/parquet/      Partitioned Parquet
data/processed/    Splits, ID mappings, cold-start embeddings
src/data/          Pipeline: csv_to_parquet, splits, dataset, cold_start
src/models/        two_tower, matrix_factorization, trainer, reranker
cpp/               C++ HNSW engine (header-only index + pybind11 bindings)
scripts/           train_all, build_index, evaluate, infer, generate_cold_start
configs/           data / model / retrieval / reranker YAML
app/main.py        FastAPI server
checkpoints/       Model checkpoints (gitignored, regenerable)
outputs/           Evaluation metrics
```

Checkpoints, data, and C++ build artifacts are gitignored — they're large and reproducible
from the scripts above.

## Notes

- `best.pt` (with optimizer state) is for resuming; `best_model.pt` (weights + config) is what
  `evaluate.py`, `build_index.py`, `infer.py`, and the API load. Loaders construct the right
  model class (ID-based or history-based) from the checkpoint's saved config.
- Serving retrieves 500 HNSW candidates, filters out movies the user already rated (via the
  precomputed `seen_items.npz`), and sends the top 25 to the reranker.
- `popularity_weight` in `configs/retrieval.yaml` is 0 for the served logQ-corrected model
  (any blend makes it worse); set it to ~0.1 only for checkpoints trained without
  `logq_correction`.
- The reranker is token-frugal: 25 candidates,
  title+genres only, compact index-list output capped at 512 tokens, responses disk-cached
  by prompt hash (`outputs/reranker_cache/`), and 429s retried with exponential backoff. It
  falls back to retrieval order if the LLM API still fails.
- Matrix Factorization exists as a baseline but nothing in the serving path uses it.
- Unit tests (`PYTHONPATH=. python3 -m pytest tests/ -q`) cover metadata alignment,
  history pooling/exclusion/ID-dropout semantics, and reranker response parsing.
