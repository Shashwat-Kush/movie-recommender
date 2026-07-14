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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerWithMetadata
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


def load_full_movie_mapping(splits_dir: Path) -> Dict[int, int]:
    corpus = pd.read_parquet(splits_dir / "cold_start_corpus.parquet")
    return {i: int(corpus.iloc[i]["movieId"]) for i in range(len(corpus))}


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
    n_users: int,
    n_items: int,
    metadata_dim: int,
    device: torch.device,
) -> TwoTowerWithMetadata:
    if "model" not in _model_cache:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ckpt_config = checkpoint.get("config", {})
        ckpt_n_items = ckpt_config.get("n_items", n_items)
        
        model = TwoTowerWithMetadata(
            n_users=n_users,
            n_items=ckpt_n_items,
            metadata_dim=metadata_dim,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=128,
            dropout=0.1,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        
        if n_items != ckpt_n_items:
            with torch.no_grad():
                new_item_emb = nn.Embedding(n_items, 128)
                nn.init.normal_(new_item_emb.weight, std=0.01)
                new_item_emb.weight[:ckpt_n_items] = model.item_embedding.weight
                model.item_embedding = new_item_emb
                model.n_items = n_items
        
        model.to(device)
        model.eval()
        _model_cache["model"] = model
    return _model_cache["model"]


def get_query_embedding(
    model: TwoTowerWithMetadata,
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

    full_idx_to_movie_id = load_full_movie_mapping(splits_dir)
    print(f"Loaded {len(full_idx_to_movie_id):,} movies (full corpus)")

    n_users = len(user_mapping)
    n_items = len(idx_to_movie_id)
    metadata_dim = 128

    checkpoint_path = Path("checkpoints/two_tower/best_model.pt")
    load_query_embedding_model(checkpoint_path, n_users, n_items, metadata_dim, device)

    _model_cache["user_mapping"] = user_mapping
    _model_cache["idx_to_movie_id"] = idx_to_movie_id
    _model_cache["full_idx_to_movie_id"] = full_idx_to_movie_id
    _model_cache["device"] = device
    _model_cache["splits_dir"] = splits_dir

    print("API server ready!")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    user_mapping = _model_cache["user_mapping"]
    full_idx_to_movie_id = _model_cache["full_idx_to_movie_id"]
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
        k=50,
        ef_search=100,
    )

    candidate_movie_ids = [full_idx_to_movie_id[int(idx)] for idx in indices]

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