# movie-recommender

A MovieLens 25M recommendation engine built to run on an 8GB M2 Mac: streaming data
pipeline → Two-Tower retrieval (PyTorch/MPS) → custom C++ HNSW index → Groq LLM reranking,
served over FastAPI.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

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
                            Groq Llama-3 reranker (top-N)
                                          ▼
                                  FastAPI /recommend
```

## Setup

```bash
pip install -r requirements.txt

# Groq API key for the reranker
echo 'GROQ_API_KEY=your_key_here' > .env

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

# 3. Train Two-Tower  (writes checkpoints/two_tower/best_model.pt)
PYTHONPATH=. python3 -u scripts/train_all.py --model two_tower

# 4. Build the HNSW index from the trained model
PYTHONPATH=. python3 scripts/build_index.py --checkpoint checkpoints/two_tower/best_model.pt

# 5. Evaluate  (writes outputs/eval_loo.json)
PYTHONPATH=. python3 scripts/evaluate.py

# 6. Serve
python3 -m uvicorn app.main:app --port 8000
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "query": "funny action movies", "top_k": 5}'
```

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
  `evaluate.py`, `build_index.py`, `infer.py`, and the API load.
- Matrix Factorization exists as a baseline but nothing in the serving path uses it.
- The reranker falls back to raw HNSW ordering if Groq fails or returns unparseable JSON.
