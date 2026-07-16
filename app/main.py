"""FastAPI server for movie recommendations."""
import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.models.reranker import LLMReranker, create_reranker


app = FastAPI(title="Movie Recommender API", version="1.0.0")


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID for personalization")
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations")


class ColdRecommendRequest(BaseModel):
    liked_movie_ids: List[int] = Field(..., min_length=1, max_length=20, description="Movies the user picked as liked, most-liked first")
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations")


class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    rerank_score: float
    retrieval_rank: Optional[int] = None  # 1-based position in the retrieval candidate list
    distance: Optional[float] = None  # cosine distance to the user embedding (lower = closer)


class Candidate(BaseModel):
    movieId: int
    title: str
    genres: str
    retrieval_rank: int
    distance: float


class Timing(BaseModel):
    hnsw_ms: float
    rerank_ms: float


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached: bool


class RecommendResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    candidates: List[Candidate]
    timing: Timing
    usage: Optional[Usage] = None


_config_cache: Dict[str, Any] = {}
_model_cache: Dict[str, Any] = {}


def load_config(config_path: str) -> dict:
    if config_path not in _config_cache:
        with open(config_path, "r") as f:
            _config_cache[config_path] = yaml.safe_load(f)
    return _config_cache[config_path]


def load_user_mapping(mapping_path: Path) -> Dict[int, int]:
    table = pq.read_table(mapping_path)
    return dict(zip(table.column("userId").to_pylist(), table.column("user_idx").to_pylist()))


def load_movie_mapping(mapping_path: Path) -> Tuple[Dict[int, int], Dict[int, int]]:
    table = pq.read_table(mapping_path)
    movie_ids = table.column("movieId").to_pylist()
    movie_idxs = table.column("movie_idx").to_pylist()
    id_to_idx = dict(zip(movie_ids, movie_idxs))
    idx_to_id = dict(zip(movie_idxs, movie_ids))
    return id_to_idx, idx_to_id


def load_movie_metadata(movies_parquet_dir: Path, movie_ids: List[int]) -> List[Dict[str, Any]]:
    dataset = ds.dataset(movies_parquet_dir, format="parquet")
    scanner = dataset.scanner(
        columns=["movieId", "title", "genres"],
        filter=ds.field("movieId").isin(movie_ids),
    )
    table = scanner.to_table()
    df = table.to_pandas()
    df = df.set_index("movieId")

    results = []
    for mid in movie_ids:
        if mid in df.index:
            row = df.loc[mid]
            results.append({
                "movieId": int(mid),
                "title": str(row["title"]) if pd.notna(row["title"]) else "",
                "genres": str(row["genres"]) if pd.notna(row["genres"]) else "",
                "tags": "",
            })
        else:
            results.append({"movieId": int(mid), "title": "", "genres": "", "tags": ""})
    return results


def load_query_embedding_model(
    checkpoint_path: Path,
    n_items: int,
    device: torch.device,
):
    if "model" not in _model_cache:
        # Constructs the right model class (ID-based or history-based) from
        # checkpoint config.
        from scripts.evaluate import load_model

        model, ckpt_config = load_model(checkpoint_path, device)
        ckpt_n_items = ckpt_config.get("n_items", n_items)
        if ckpt_n_items != n_items:
            raise RuntimeError(
                f"Checkpoint has {ckpt_n_items} items but movie_mapping.parquet has {n_items}. "
                "The mapping changed since training — retrain rather than reshaping the "
                "embedding table, which would desync it from the HNSW index."
            )
        _model_cache["model"] = model
    return _model_cache["model"]


def get_query_embedding(
    model,
    user_id: int,
    user_mapping: Dict[int, int],
    device: torch.device,
) -> np.ndarray:
    if user_id not in user_mapping:
        raise ValueError(f"User {user_id} not found in mapping")
    internal_id = user_mapping[user_id]
    user_tensor = torch.tensor([internal_id], dtype=torch.long, device=device)
    with torch.no_grad():
        emb = model.get_user_embeddings(user_tensor)
        emb = emb.cpu().numpy().astype(np.float32)
    return emb[0]


def search_hnsw(
    index_path: Path,
    query_vector: np.ndarray,
    k: int = 50,
    ef_search: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    cpp_build = Path(__file__).parent.parent / "cpp" / "build_arm64"
    sys.path.insert(0, str(cpp_build))
    try:
        import _cpp
    except ImportError:
        raise RuntimeError("C++ engine not built. Run: cd cpp && ./build.sh")
    config = _cpp.RetrievalConfig()
    engine = _cpp.RetrievalEngine(config)
    engine.load(str(index_path), config)
    indices, distances = engine.search(query_vector, k=k, ef_search=ef_search)
    return np.array(indices), np.array(distances)


def get_seen_movie_ids(splits_dir: Path, user_id: int) -> set:
    """MovieIds the user already rated (LOO train+val) — never re-recommend these."""
    seen = set()
    for name in ("train_loo.parquet", "val_loo.parquet"):
        path = splits_dir / name
        if path.exists():
            table = ds.dataset(path).scanner(
                columns=["movieId"], filter=ds.field("userId") == user_id
            ).to_table()
            seen.update(table.column("movieId").to_pylist())
    return seen


def load_reranker_config(config_path: str = "configs/reranker.yaml") -> dict:
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


@app.on_event("startup")
async def startup():
    global _model_cache, _config_cache
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_config = load_config("configs/data.yaml")
    splits_dir = Path(data_config["splits"]["output_dir"])

    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    print(f"Loaded {len(user_mapping):,} users")

    movie_id_to_idx, idx_to_movie_id = load_movie_mapping(splits_dir / "movie_mapping.parquet")
    print(f"Loaded {len(idx_to_movie_id):,} movies (training)")
    _model_cache["movie_id_to_idx"] = movie_id_to_idx

    # Popularity blend (see configs/retrieval.yaml): rank candidates by
    # cosine + popularity_weight * log(pop_count + 1). Counts are written by
    # scripts/evaluate.py in movie_map index order.
    retrieval_cfg = load_config("configs/retrieval.yaml") if Path("configs/retrieval.yaml").exists() else {}
    _model_cache["popularity_weight"] = float(retrieval_cfg.get("popularity_weight", 0.0))
    _model_cache["candidate_pool"] = int(retrieval_cfg.get("candidate_pool", 500))
    counts_path = splits_dir / "popularity_counts.npy"
    if counts_path.exists():
        _model_cache["log_pop"] = np.log(np.load(counts_path) + 1.0)
    else:
        _model_cache["log_pop"] = None
        if _model_cache["popularity_weight"] > 0:
            print(f"Warning: {counts_path} not found (run scripts/evaluate.py); popularity blend disabled")

    # Precomputed per-user seen movieIds (written by scripts/evaluate.py) — avoids
    # a 20M-row parquet scan on the first request for each user.
    seen_path = splits_dir / "seen_items.npz"
    if seen_path.exists():
        seen_npz = np.load(seen_path)
        _model_cache["seen_table"] = (seen_npz["offsets"], seen_npz["movie_ids"])
        print(f"Loaded seen-items table ({len(seen_npz['movie_ids']):,} interactions)")
    else:
        _model_cache["seen_table"] = None
        print(f"Note: {seen_path} not found (run scripts/evaluate.py); falling back to per-request parquet scans")

    checkpoint_path = Path("checkpoints/two_tower_history_v2/best_model.pt")
    load_query_embedding_model(checkpoint_path, len(idx_to_movie_id), device)

    _model_cache["user_mapping"] = user_mapping
    _model_cache["idx_to_movie_id"] = idx_to_movie_id
    _model_cache["device"] = device
    _model_cache["splits_dir"] = splits_dir

    print("API server ready!")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    user_mapping = _model_cache["user_mapping"]
    device = _model_cache["device"]
    splits_dir = _model_cache["splits_dir"]
    model = _model_cache["model"]

    try:
        query_vec = get_query_embedding(model, request.user_id, user_mapping, device)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Cached per user: the splits are static offline data, so no invalidation needed.
    seen_cache = _model_cache.setdefault("seen_cache", {})
    if request.user_id not in seen_cache:
        seen_table = _model_cache.get("seen_table")
        if seen_table is not None:
            offsets, movie_ids = seen_table
            uidx = user_mapping[request.user_id]
            seen_cache[request.user_id] = set(movie_ids[offsets[uidx] : offsets[uidx + 1]].tolist())
        else:
            seen_cache[request.user_id] = get_seen_movie_ids(splits_dir, request.user_id)

    return _run_pipeline(query_vec, seen_cache[request.user_id], request.query, request.top_k)


@app.post("/recommend_cold", response_model=RecommendResponse)
async def recommend_cold(request: ColdRecommendRequest):
    """Recommendations for a brand-new user from a handful of picked movies.

    No user ID or stored history: the picked movies are pooled through the history
    tower (same recency weighting as stored histories) to form the user embedding.
    """
    movie_id_to_idx = _model_cache["movie_id_to_idx"]
    device = _model_cache["device"]
    model = _model_cache["model"]

    item_idxs = [movie_id_to_idx[mid] for mid in request.liked_movie_ids if mid in movie_id_to_idx]
    if not item_idxs:
        raise HTTPException(status_code=404, detail="None of the picked movieIds are in the catalog")

    with torch.no_grad():
        query_vec = model.get_user_embedding_from_items(
            torch.tensor(item_idxs, dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)[0]

    # Never re-recommend what the user just told us they've watched.
    return _run_pipeline(query_vec, set(request.liked_movie_ids), request.query, request.top_k)


def _run_pipeline(query_vec: np.ndarray, seen: set, query: str, top_k: int) -> RecommendResponse:
    """Shared retrieve -> blend -> filter -> rerank pipeline for both endpoints."""
    idx_to_movie_id = _model_cache["idx_to_movie_id"]

    t0 = time.perf_counter()
    indices, distances = search_hnsw(
        Path("cpp/build_arm64/index.bin"),
        query_vec,
        k=_model_cache["candidate_pool"],
        ef_search=100,
    )
    hnsw_ms = (time.perf_counter() - t0) * 1000

    # Engine distance = -dot of L2-normalized vectors; report proper cosine
    # distance (1 - cosine, lower = closer).
    cos_dist_by_idx = {int(i): float(1.0 + d) for i, d in zip(indices, distances)}

    # Blend popularity into retrieval scores (engine distance = -dot).
    log_pop = _model_cache["log_pop"]
    w = _model_cache["popularity_weight"]
    if log_pop is not None and w > 0:
        scores = -distances + w * log_pop[indices]
        indices = indices[np.argsort(-scores)]

    # Index positions are movie_map indices (see scripts/build_index.py).
    # Drop seen movies, then keep the reranker prompt budget: the tail of the
    # list is not where reranking earns anything.
    candidate_ids, candidate_dists = [], []
    for idx in indices:
        mid = idx_to_movie_id[int(idx)]
        if mid not in seen:
            candidate_ids.append(mid)
            candidate_dists.append(cos_dist_by_idx[int(idx)])
        if len(candidate_ids) == 25:
            break

    movies_meta = load_movie_metadata(Path("data/parquet/movies"), candidate_ids)
    rank_by_id = {mid: r + 1 for r, mid in enumerate(candidate_ids)}
    dist_by_id = dict(zip(candidate_ids, candidate_dists))

    reranker = create_reranker(load_reranker_config("configs/reranker.yaml"))
    t1 = time.perf_counter()
    reranked = reranker.rerank(query, movies_meta, top_k=top_k)
    rerank_ms = (time.perf_counter() - t1) * 1000

    return RecommendResponse(
        recommendations=[
            MovieRecommendation(
                movieId=m["movieId"],
                title=m["title"],
                genres=m.get("genres", ""),
                rerank_score=m.get("rerank_score", 0.0),
                retrieval_rank=rank_by_id.get(m["movieId"]),
                distance=dist_by_id.get(m["movieId"]),
            )
            for m in reranked
        ],
        candidates=[
            Candidate(
                movieId=meta["movieId"],
                title=meta["title"],
                genres=meta.get("genres", ""),
                retrieval_rank=rank_by_id[meta["movieId"]],
                distance=dist_by_id[meta["movieId"]],
            )
            for meta in movies_meta
        ],
        timing=Timing(hnsw_ms=round(hnsw_ms, 1), rerank_ms=round(rerank_ms, 1)),
        usage=Usage(**reranker.last_usage) if reranker.last_usage else None,
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)