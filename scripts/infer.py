#!/usr/bin/env python
"""Inference pipeline: Query → Two-Tower → HNSW → Groq Rerank → Top-5."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import yaml
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
import pandas as pd

from src.models.two_tower import TwoTowerWithMetadata
from src.models.reranker import GroqReranker, create_reranker


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_user_mapping(mapping_path: Path) -> Dict[int, int]:
    """Load user_id -> internal_index mapping from parquet."""
    table = pq.read_table(mapping_path)
    return dict(zip(table.column("userId").to_pylist(), table.column("user_idx").to_pylist()))


def load_movie_mapping(mapping_path: Path) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Load movie_id <-> internal_index mappings from parquet."""
    table = pq.read_table(mapping_path)
    movie_ids = table.column("movieId").to_pylist()
    movie_idxs = table.column("movie_idx").to_pylist()
    id_to_idx = dict(zip(movie_ids, movie_idxs))
    idx_to_id = dict(zip(movie_idxs, movie_ids))
    return id_to_idx, idx_to_id


def load_movie_metadata(movies_parquet_dir: Path, movie_ids: List[int]) -> List[Dict[str, Any]]:
    """Load movie metadata (title, genres, tags) for given movie IDs."""
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
    """Load trained TwoTowerWithMetadata model for query embedding."""
    model = TwoTowerWithMetadata(
        n_users=n_users,
        n_items=n_items,
        metadata_dim=metadata_dim,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=128,
        dropout=0.1,
    )

    state = torch.load(checkpoint_path, map_location=device)
    # Handle both formats: direct state_dict or checkpoint dict with 'model_state_dict'
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_query_embedding(
    model: TwoTowerWithMetadata,
    user_id: int,
    user_mapping: Dict[int, int],
    device: torch.device,
) -> np.ndarray:
    """Project user ID into 128-dim query vector."""
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
    """Search HNSW index using C++ engine via pybind11."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "cpp" / "build_arm64"))

    try:
        import _cpp
    except ImportError:
        raise RuntimeError(
            "C++ engine not built. Run: cd cpp && ./build.sh\n"
            "Ensure pybind11 is installed and Python bindings are compiled."
        )

    config = _cpp.RetrievalConfig()
    engine = _cpp.RetrievalEngine(config)
    engine.load(str(index_path), config)

    indices, distances = engine.search(query_vector, k=k, ef_search=ef_search)
    return np.array(indices), np.array(distances)


def load_reranker_config(config_path: str = "configs/reranker.yaml") -> dict:
    """Load reranker configuration."""
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Movie recommendation inference pipeline")
    parser.add_argument("--query", type=str, required=True, help="User query text")
    parser.add_argument("--user-id", type=int, required=True, help="User ID for personalization")
    parser.add_argument("--top-k", type=int, default=5, help="Final number of recommendations")
    parser.add_argument("--retrieval-k", type=int, default=50, help="HNSW retrieval candidates")
    parser.add_argument("--ef-search", type=int, default=100, help="HNSW ef_search parameter")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/two_tower/best_model.pt",
        help="Path to Two-Tower checkpoint",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="cpp/build_arm64/index.bin",
        help="Path to HNSW index",
    )
    parser.add_argument(
        "--user-mapping",
        type=str,
        default="data/processed/user_mapping.parquet",
        help="Path to user ID mapping",
    )
    parser.add_argument(
        "--movie-mapping",
        type=str,
        default="data/processed/movie_mapping.parquet",
        help="Path to movie ID mapping",
    )
    parser.add_argument(
        "--movies-parquet",
        type=str,
        default="data/parquet/movies",
        help="Path to movies parquet directory",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config",
    )
    parser.add_argument(
        "--reranker-config",
        type=str,
        default="configs/reranker.yaml",
        help="Path to reranker config",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_config = load_config(args.data_config)
    splits_dir = Path(data_config["splits"]["output_dir"])

    print("Loading user mapping...")
    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    print(f"  Loaded {len(user_mapping):,} users")

    print("Loading movie mapping...")
    _, idx_to_movie_id = load_movie_mapping(splits_dir / "movie_mapping.parquet")
    print(f"  Loaded {len(idx_to_movie_id):,} movies (training)")

    # For HNSW index (built on all 27,278 movies), load full mapping from cold start corpus
    import pandas as pd
    corpus = pd.read_parquet(splits_dir / "cold_start_corpus.parquet")
    full_idx_to_movie_id = {i: int(corpus.iloc[i]["movieId"]) for i in range(len(corpus))}
    print(f"  Loaded {len(full_idx_to_movie_id):,} movies (full corpus for HNSW)")

    n_users = len(user_mapping)
    n_items_model = 26744  # matches Two-Tower checkpoint item_embedding size

    print("Loading Two-Tower model...")
    metadata_dim = 128
    model = load_query_embedding_model(
        Path(args.checkpoint),
        n_users=n_users,
        n_items=n_items_model,
        metadata_dim=metadata_dim,
        device=device,
    )

    print(f"Encoding query for user {args.user_id}...")
    query_vec = get_query_embedding(model, args.user_id, user_mapping, device)
    print(f"  Query vector shape: {query_vec.shape}")

    print(f"Searching HNSW index (top-{args.retrieval_k})...")
    indices, distances = search_hnsw(
        Path(args.index),
        query_vec,
        k=args.retrieval_k,
        ef_search=args.ef_search,
    )
    print(f"  Retrieved {len(indices)} candidates")

    candidate_movie_ids = [full_idx_to_movie_id[int(idx)] for idx in indices]

    print("Loading movie metadata...")
    movies_meta = load_movie_metadata(Path(args.movies_parquet), candidate_movie_ids)
    print(f"  Loaded metadata for {len(movies_meta)} movies")

    print("Initializing Groq reranker...")
    reranker_config = load_reranker_config(args.reranker_config)
    reranker = create_reranker(reranker_config)

    print(f"Reranking with query: '{args.query}'...")
    reranked = reranker.rerank(args.query, movies_meta, top_k=args.top_k)

    print(f"\n=== Top {args.top_k} Recommendations ===")
    for i, movie in enumerate(reranked, 1):
        score = movie.get("rerank_score", 0)
        print(f"{i}. [{movie['movieId']}] {movie['title']} | Score: {score:.1f}")
        if movie.get("genres"):
            print(f"   Genres: {movie['genres']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())