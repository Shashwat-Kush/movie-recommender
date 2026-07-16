"""FastAPI server for movie recommendations."""
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

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

from src.models.reranker import GroqReranker, create_reranker


app = FastAPI(title="Movie Recommender API", version="1.0.0")


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID for personalization")
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations")


class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    rerank_score: float


class RecommendResponse(BaseModel):
    recommendations: List[MovieRecommendation]


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

    _, idx_to_movie_id = load_movie_mapping(splits_dir / "movie_mapping.parquet")
    print(f"Loaded {len(idx_to_movie_id):,} movies (training)")

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
    idx_to_movie_id = _model_cache["idx_to_movie_id"]
    device = _model_cache["device"]
    splits_dir = _model_cache["splits_dir"]
    model = _model_cache["model"]

    try:
        query_vec = get_query_embedding(model, request.user_id, user_mapping, device)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    indices, distances = search_hnsw(
        Path("cpp/build_arm64/index.bin"),
        query_vec,
        k=_model_cache["candidate_pool"],
        ef_search=100,
    )

    # Blend popularity into retrieval scores (engine distance = -dot).
    log_pop = _model_cache["log_pop"]
    w = _model_cache["popularity_weight"]
    if log_pop is not None and w > 0:
        scores = -distances + w * log_pop[indices]
        indices = indices[np.argsort(-scores)]

    # Index positions are movie_map indices (see scripts/build_index.py).
    # Drop movies the user already rated, then keep the reranker prompt budget.
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
    seen = seen_cache[request.user_id]
    candidate_movie_ids = [
        mid for idx in indices
        if (mid := idx_to_movie_id[int(idx)]) not in seen
    ][:25]  # reranker token budget: the tail of the list is not where reranking earns anything

    movies_meta = load_movie_metadata(Path("data/parquet/movies"), candidate_movie_ids)

    reranker_config = load_reranker_config("configs/reranker.yaml")
    reranker = create_reranker(reranker_config)

    reranked = reranker.rerank(request.query, movies_meta, top_k=request.top_k)

    recommendations = [
        MovieRecommendation(
            movieId=movie["movieId"],
            title=movie["title"],
            genres=movie.get("genres", ""),
            rerank_score=movie.get("rerank_score", 0.0),
        )
        for movie in reranked
    ]

    return RecommendResponse(recommendations=recommendations)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)