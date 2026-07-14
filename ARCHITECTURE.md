# MovieLens 25M Recommendation Engine — Architectural Plan

**Target:** M2 Mac, 8GB unified memory  
**Dataset:** MovieLens 25M (ratings.csv ~600MB, movies.csv, tags.csv, genome-scores.csv, genome-tags.csv, links.csv)  
**Embedding dim:** 128 | **Top-K retrieval:** 500 | **Reranker:** Groq API (Llama-3) | **Eval:** Time-split + Leave-one-out

---

## Directory Structure

```
movie-recommender/
├── data/
│   ├── raw/                    # Original CSVs (gitignored)
│   ├── parquet/                # Partitioned Parquet outputs
│   │   ├── ratings/
│   │   │   ├── part-000.parquet
│   │   │   └── ...
│   │   ├── movies/
│   │   ├── tags/
│   │   ├── genome_scores/
│   │   └── genome_tags/
│   └── processed/              # Training/validation/test splits
│       ├── train.parquet
│       ├── val_time.parquet
│       ├── val_loo.parquet
│       └── test.parquet
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── csv_to_parquet.py          # Stage 1: Chunked CSV → Parquet
│   │   ├── dataset.py                 # IterableDataset for streaming
│   │   ├── splits.py                  # Time-split & LOO split logic
│   │   └── cold_start.py              # Stage 3: Metadata embedding pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative.py           # Stage 2: Item-item CF baseline
│   │   ├── matrix_factorization.py    # Stage 2: ALS/SGD MF
│   │   ├── two_tower.py               # Stage 2: Two-tower retrieval model
│   │   ├── trainer.py                 # Training loop with MPS + OOM guards
│   │   └── checkpoint.py              # Sharded checkpoint save/load
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── cpp_bridge.py              # Python ↔ C++ bindings (pybind11)
│   │   ├── index_builder.py           # Build HNSW/IVF index from embeddings
│   │   └── search.py                  # Query interface
│   ├── reranker/
│   │   ├── __init__.py
│   │   ├── groq_client.py             # Stage 5: Groq API client with batching
│   │   ├── prompt_templates.py        # Reranking prompts
│   │   └── scorer.py                  # Score + reorder candidates
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # NDCG@K, Recall@K, MRR
│   │   ├── simulator.py               # Stage 6: A/B simulator
│   │   ├── business_objective.py      # Revenue/engagement simulation
│   │   └── ab_framework.py            # Experiment assignment + analysis
│   └── serving/
│       ├── __init__.py
│       ├── pipeline.py                # End-to-end inference pipeline
│       └── api.py                     # FastAPI endpoint (optional)
├── cpp/
│   ├── include/
│   │   ├── retrieval_engine.hpp       # Public C++ API
│   │   ├── hnsw_index.hpp             # HNSW index structure
│   │   ├── simd_dot.hpp               # AVX2/NEON dot product kernels
│   │   └── memory_pool.hpp            # Arena allocator for 8GB budget
│   ├── src/
│   │   ├── retrieval_engine.cpp       # Main engine logic
│   │   ├── hnsw_index.cpp             # HNSW implementation
│   │   ├── simd_dot.cpp               # Vectorized dot products
│   │   ├── memory_pool.cpp            # Pool allocator
│   │   └── pybind_module.cpp          # Python bindings entry point
│   ├── CMakeLists.txt
│   └── build.sh                       # Build script (Release, -O3 -march=native)
├── scripts/
│   ├── download_data.sh               # Fetch MovieLens 25M
│   ├── run_pipeline.py                # Orchestrate all stages
│   ├── train_all.py                   # Train CF → MF → Two-Tower
│   ├── build_index.py                 # Build C++ index from embeddings
│   ├── generate_cold_start.py         # Offline metadata embeddings
│   ├── evaluate.py                    # Run A/B simulator
│   └── serve.py                       # Start inference server
├── configs/
│   ├── data.yaml                      # Paths, chunk sizes, partitions
│   ├── model.yaml                     # Dim=128, layers, lr, batch sizes
│   ├── retrieval.yaml                 # HNSW params (M, efConstruction, efSearch)
│   ├── reranker.yaml                  # Groq model, batch size, timeout
│   └── eval.yaml                      # Split dates, K values, business weights
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_retrieval.py
│   ├── test_reranker.py
│   └── test_evaluation.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Data Flow Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Raw CSVs       │────▶│  Chunked Parquet │────▶│  Train/Val/Test     │
│  (data/raw/)    │     │  (data/parquet/) │     │  (data/processed/)  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                             │
                    ┌────────────────────────────────────────┼─────────────────────┐
                    ▼                                        ▼                     ▼
           ┌─────────────────┐                     ┌─────────────────┐   ┌─────────────────┐
           │ Collaborative   │                     │ Matrix          │   │ Two-Tower       │
           │ Filtering       │                     │ Factorization   │   │ (PyTorch MPS)   │
           │ (Item-Item)     │                     │ (ALS/SGD)       │   │                 │
           └────────┬────────┘                     └────────┬────────┘   └────────┬────────┘
                    │                                       │                     │
                    └───────────────────────┬───────────────┴─────────────────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │ User/Item       │
                                   │ Embeddings      │
                                   │ (128-dim)       │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ C++ HNSW Index  │
                                   │ (cpp/build/)    │
                                   └────────┬────────┘
                                            │
                                            ▼
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
           ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
           │ Cold-Start      │     │ Main Retrieval  │     │ Reranker        │
           │ Pipeline        │     │ (Top-500 HNSW)  │     │ (Groq Llama-3)  │
           │ (MiniLM-L6-v2)  │     │                 │     │                 │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │ Final Ranked    │
                                   │ Recommendations │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ Evaluation      │
                                   │ Simulator       │
                                   │ (NDCG, Business)│
                                   └─────────────────┘
```

---

## Stage 1: Data Pipeline — CSV → Parquet (Memory-Safe)

### Goals
- Stream 600MB ratings.csv without loading full DataFrame
- Partition by `userId` hash for co-located user interactions
- Write Parquet with ZSTD compression
- Produce train/val/test splits (time-based + LOO)

### Files to Write

| File | Responsibility |
|------|----------------|
| `src/data/csv_to_parquet.py` | Chunked reader (100k rows/chunk), type coercion, Parquet writer |
| `src/data/dataset.py` | `IterableDataset` yielding (user, item, rating, timestamp) tuples |
| `src/data/splits.py` | `time_split(cutoff_date)`, `leave_one_out_per_user()` |
| `configs/data.yaml` | `chunk_size: 100000`, `partition_cols: ["user_bucket"]`, `num_buckets: 256` |

### Memory Guardrails
- `pd.read_csv(chunksize=100_000)` — never materialize full DataFrame
- `pyarrow.Table.from_pandas(chunk).to_parquet()` — stream to disk
- Explicit `del chunk; gc.collect()` per iteration
- Process ratings first (largest), then movies/tags/genome (small, can load fully)

### Output
```
data/parquet/ratings/part-000.parquet ... part-249.parquet  (~250 files × ~2.5MB)
data/processed/train.parquet
data/processed/val_time.parquet   (interactions after 2019-01-01)
data/processed/val_loo.parquet    (last interaction per user)
data/processed/test.parquet       (held-out for final eval)
```

---

## Stage 2: Model Training — PyTorch MPS with OOM Guards

### Models to Implement

| Model | File | Key Details |
|-------|------|-------------|
| Item-Item CF | `src/models/collaborative.py` | Cosine similarity on co-occurrence, top-K neighbors per item |
| Matrix Factorization | `src/models/matrix_factorization.py` | ALS (implicit) + SGD (explicit), 128 factors, λ=0.01 |
| Two-Tower | `src/models/two_tower.py` | User tower: 3-layer MLP(128→256→128), Item tower: same + metadata concat |

### Training Infrastructure

| File | Responsibility |
|------|----------------|
| `src/models/trainer.py` | `train_epoch()`, `validate()`, MPS device placement, gradient accumulation |
| `src/models/checkpoint.py` | Sharded save (user_emb.pt, item_emb.pt, model.pt), resume support |
| `configs/model.yaml` | `batch_size: 2048`, `accum_steps: 4`, `lr: 1e-3`, `weight_decay: 1e-5`, `max_epochs: 10` |

### MPS Memory Management (Critical for 8GB)

```python
# In trainer.py — enforce at every step
def train_step(batch):
    with torch.mps.autocast(dtype=torch.float16):  # FP16 on MPS
        loss = model(batch)
    scaler.scale(loss).backward()
    if (step + 1) % accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    # Explicit cleanup
    del batch, loss
    torch.mps.empty_cache()
    gc.collect()
```

- Batch size 2048 with grad accum 4 → effective 8192
- FP16 autocast + gradient scaling
- `torch.mps.empty_cache()` + `gc.collect()` every step
- No `DataLoader` pin_memory (MPS doesn't benefit)
- Single worker (`num_workers=0`) to avoid multiprocessing overhead

### Output
```
checkpoints/
├── collaborative/
│   └── item_similarities.npy       # (n_items, top_k)
├── matrix_factorization/
│   ├── user_factors.npy            # (n_users, 128)
│   └── item_factors.npy            # (n_items, 128)
└── two_tower/
    ├── user_tower.pt
    ├── item_tower.pt
    ├── user_embeddings.npy         # (n_users, 128)
    └── item_embeddings.npy         # (n_items, 128)
```

---

## Stage 3: RAG Cold-Start Pipeline — Metadata Embeddings

### Goal
Generate embeddings for movies with zero ratings using metadata (title, genres, tags, genome scores).

### Files to Write

| File | Responsibility |
|------|----------------|
| `src/data/cold_start.py` | Load movies/tags/genome, build text corpus per movie |
| `scripts/generate_cold_start.py` | Offline batch encoding with `sentence-transformers/all-MiniLM-L6-v2` |

### Implementation
- Text template: `"{title}. Genres: {genres}. Tags: {top_tags}. Genome: {top_genome_tags}"`
- Batch encode 32 movies at a time on MPS (MiniLM-L6-v2 ~90MB, fits easily)
- Output: `data/processed/cold_start_embeddings.npy` (n_cold_items, 384) → project to 128 via learned linear layer
- Learn projection: `Linear(384, 128)` trained on items with both CF + text embeddings (MSE loss)

### Memory
- MiniLM-L6-v2: 22M params ≈ 90MB FP32, 45MB FP16
- Process in batches of 32 → peak ~200MB
- No persistent GPU allocation — encode → save → release

---

## Stage 4: C++ Retrieval Engine — HNSW + SIMD Dot Products

### Requirements
- Build HNSW index from 128-dim item embeddings (~65k items for ML-25M)
- Top-500 retrieval < 5ms p99 on M2
- Custom arena allocator to bound memory
- pybind11 bindings for Python inference pipeline

### C++ Files

| File | Responsibility |
|------|----------------|
| `cpp/include/retrieval_engine.hpp` | C API: `build_index()`, `search()`, `free_index()` |
| `cpp/include/hnsw_index.hpp` | HNSW graph structure, layer assignment, neighbor lists |
| `cpp/include/simd_dot.hpp` | `dot_f32_avx2()`, `dot_f32_neon()` — 128-dim unrolled |
| `cpp/include/memory_pool.hpp` | Arena allocator (pre-allocate 2GB max), no malloc in hot path |
| `cpp/src/retrieval_engine.cpp` | Orchestrate build + search, manage pool lifetime |
| `cpp/src/hnsw_index.cpp` | Insert, search with efConstruction/efSearch |
| `cpp/src/simd_dot.cpp` | Vectorized kernels, fallback to scalar |
| `cpp/src/memory_pool.cpp` | Pool init, alloc, reset, destruct |
| `cpp/src/pybind_module.cpp` | `py::class_<RetrievalEngine>`, numpy array binding (zero-copy) |
| `cpp/CMakeLists.txt` | `Release`, `-O3 -march=native -ffast-math`, link pybind11 |
| `cpp/build.sh` | `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` |

### HNSW Parameters (tuned for 65k vectors, 128-dim)
- `M = 32` (max connections per node)
- `efConstruction = 200`
- `efSearch = 100` (adjustable at query time)
- `max_elements = 70000` (headroom)

### Memory Budget (C++ side)
- Vectors: 70k × 128 × 4B = ~36 MB
- HNSW graph: ~70k × 32 × 4B × 2 (bidirectional) = ~18 MB
- Arena pool: 2 GB reserved (configurable)
- Total C++ footprint: < 100 MB

### Python Bridge (`src/retrieval/cpp_bridge.py`)
```python
class RetrievalEngine:
    def __init__(self, index_path: str):
        self._engine = _cpp.RetrievalEngine(index_path)
    def search(self, query: np.ndarray, k: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        # query: (128,) float32 → returns (indices, distances) both (k,)
        return self._engine.search(query, k)
```

---

## Stage 5: LLM Reranker — Groq API (Llama-3)

### Files to Write

| File | Responsibility |
|------|----------------|
| `src/reranker/groq_client.py` | Async HTTP client, retry/backoff, batch prompts |
| `src/reranker/prompt_templates.py` | System + user templates for movie reranking |
| `src/reranker/scorer.py` | Parse LLM output → relevance scores → reorder |

### Prompt Strategy
```
System: "You are a movie recommendation expert. Rank these candidates for the user."
User: "User history: [top-5 watched movies with ratings]. Candidates: [500 movie titles + genres]. Return top-20 as JSON: [movie_id, score]."
```

### Batching & Latency
- Send 50 candidates per request (Groq context window)
- Parallelize 10 requests → 500 candidates in ~2-3s
- Cache reranker results per (user, candidate_set_hash) for 1hr
- Fallback: if Groq fails, return HNSW scores

### Output
`reranked_top20: List[(movie_id, score)]`

---

## Stage 6: Evaluation Simulator — A/B Testing + Business Objective

### Files to Write

| File | Responsibility |
|------|----------------|
| `src/evaluation/metrics.py` | `ndcg_at_k()`, `recall_at_k()`, `mrr_at_k()`, `coverage()` |
| `src/evaluation/business_objective.py` | Simulated revenue: `sum(score * price * conversion_prob)` |
| `src/evaluation/ab_framework.py` | `Experiment`, `Variant`, `assign_user()`, `analyze()` |
| `src/evaluation/simulator.py` | Main loop: replay interactions, log metrics per variant |

### Evaluation Protocols

**Time-Based Split**
- Train: interactions before 2019-01-01
- Val: interactions 2019-01-01 to 2019-06-01
- Test: after 2019-06-01

**Leave-One-Out (per user)**
- Train: all but last interaction
- Val: last interaction (held-out)
- Test: separate holdout

### A/B Variants to Compare
| Variant | Retrieval | Reranker |
|---------|-----------|----------|
| A (baseline) | Item-Item CF (top-500) | None |
| B | Matrix Factorization (top-500) | None |
| C | Two-Tower HNSW (top-500) | None |
| D | Two-Tower HNSW (top-500) | Groq Llama-3 |
| E | Two-Tower + Cold-Start (hybrid) | Groq Llama-3 |

### Business Objective Function
```python
def business_value(recommendations, user_history, catalog_prices):
    # Conversion prob decays with rank position
    # Price from links.csv (TMDB) or synthetic
    # Engagement weight: rating × watch_time_proxy
    return sum(
        relevance_score(rank) * price[movie_id] * engagement_weight(user_history, movie_id)
        for rank, movie_id in enumerate(recommendations)
    )
```

### Output
```
evaluation_results/
├── time_split/
│   ├── metrics.json        # NDCG@10, Recall@20, Business $ per variant
│   └── statistical_test.json  # t-test, confidence intervals
├── loo_split/
│   ├── metrics.json
│   └── statistical_test.json
└── ab_report.md            # Human-readable summary
```

---

## Execution Sequence (Scripts)

| Order | Script | Description |
|-------|--------|-------------|
| 1 | `scripts/download_data.sh` | Fetch ML-25M to `data/raw/` |
| 2 | `python -m src.data.csv_to_parquet` | Stage 1: CSV → Parquet |
| 3 | `python -m src.data.splits` | Create train/val/test splits |
| 4 | `python -m src.models.trainer --model collaborative` | Train CF baseline |
| 5 | `python -m src.models.trainer --model mf` | Train Matrix Factorization |
| 6 | `python -m src.models.trainer --model two_tower` | Train Two-Tower (MPS) |
| 7 | `python scripts/generate_cold_start.py` | Stage 3: Cold-start embeddings |
| 8 | `python scripts/build_index.py` | Stage 4: Build C++ HNSW index |
| 9 | `cd cpp && ./build.sh` | Compile C++ engine |
| 10 | `python -m src.evaluation.simulator --protocol time` | Stage 6: Time-split eval |
| 11 | `python -m src.evaluation.simulator --protocol loo` | Stage 6: LOO eval |
| 12 | `python scripts/serve.py` | Optional: Start inference API |

---

## Memory Budget Summary (8GB Unified)

| Component | Est. Peak Memory |
|-----------|------------------|
| OS + Python baseline | ~2.5 GB |
| PyTorch MPS (model + grads + optimizer) | ~1.5 GB |
| DataLoader buffers (streaming) | ~200 MB |
| MiniLM-L6-v2 (cold-start, transient) | ~200 MB |
| C++ engine (loaded at inference) | ~100 MB |
| Groq API client (network buffers) | ~50 MB |
| **Headroom** | **~3.5 GB** |

### OOM Prevention Checklist
- [ ] `torch.mps.empty_cache()` + `gc.collect()` every training step
- [ ] `batch_size=2048`, `accum_steps=4`, FP16 autocast
- [ ] Chunked Parquet reads — never `pd.read_csv()` full file
- [ ] C++ arena allocator — no `new`/`malloc` in search path
- [ ] Cold-start encoding: batch=32, release after save
- [ ] Groq: async, no local model weights

---

## Open Questions for Implementation

1. **C++ SIMD**: Target AVX2 (x86) + NEON (ARM/M2)? M2 is ARM64 → NEON only. Confirm `simd_dot_neon.cpp` is primary.
2. **HNSW Library**: Custom implementation per spec, or allow `hnswlib` as dependency? Spec says "custom C++ backend" — will implement from scratch.
3. **Two-Tower Item Features**: What metadata concatenated? Genres (multi-hot), year, genome scores (top-10)? Need feature spec.
4. **Groq Rate Limits**: Batch size 50, 10 parallel → 500 req/min. Check Groq tier limits.
5. **Checkpoint Format**: `safetensors` vs `.pt`? `.pt` is fine for PyTorch-only.
6. **Serving**: FastAPI + C++ engine loaded once at startup? Or separate processes?

---

## Next Steps

Upon approval:
1. Create `configs/*.yaml` with exact hyperparameters
2. Implement Stage 1 data pipeline (highest risk for OOM)
3. Build C++ engine in parallel (independent)
4. Train models sequentially (CF → MF → Two-Tower)
5. Wire retrieval → reranker → evaluation

**Confirm plan or request modifications.**