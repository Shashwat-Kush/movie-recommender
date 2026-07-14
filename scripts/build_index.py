#!/usr/bin/env python
"""Build HNSW index from item embeddings (cold-start or trained)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add C++ build to path
cpp_build = Path(__file__).parent.parent / "cpp" / "build_arm64"
sys.path.insert(0, str(cpp_build))

try:
    import _cpp
except ImportError:
    print("Error: C++ engine not built. Run: cd cpp && ./build.sh")
    sys.exit(1)

from src.data.cold_start import build_aligned_metadata
from src.models.two_tower import TwoTowerWithMetadata


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_index(
    item_embeddings: np.ndarray,
    index_path: Path,
    config: dict = None,
):
    """Build and save HNSW index."""
    if config is None:
        config = _cpp.RETRIEVAL_DEFAULT_CONFIG
    
    print(f"Building HNSW index for {item_embeddings.shape[0]} items, dim={item_embeddings.shape[1]}")
    
    engine = _cpp.RetrievalEngine(config)
    
    # Build index
    engine.build(item_embeddings.astype(np.float32))
    
    # Save index
    engine.save(str(index_path))
    
    # Print stats
    stats = engine.get_stats()
    print(f"Index built: {stats['element_count']} elements, {stats['memory_used_bytes'] / 1024 / 1024:.1f} MB")
    
    return engine


def main():
    parser = argparse.ArgumentParser(description="Build HNSW index from item embeddings")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (optional)")
    parser.add_argument("--index-path", type=str, default="cpp/build_arm64/index.bin", help="Output index path")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config path")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Model config path")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data dir")
    parser.add_argument("--retrieval-config", type=str, default="configs/retrieval.yaml", help="Retrieval config path")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Index positions are movie_map indices, so the API can map results back through
    # movie_mapping.parquet. Metadata must be in that same order.
    processed_dir = Path(args.processed_dir)
    movie_map_table = pq.read_table(processed_dir / "movie_mapping.parquet")
    movie_map = dict(zip(
        movie_map_table.column("movieId").to_pylist(),
        movie_map_table.column("movie_idx").to_pylist(),
    ))
    item_metadata_np = build_aligned_metadata(movie_map, processed_dir)
    metadata_dim = item_metadata_np.shape[1]

    # Load retrieval config
    retrieval_config = _cpp.RETRIEVAL_DEFAULT_CONFIG
    if Path(args.retrieval_config).exists():
        ret_config = load_config(args.retrieval_config)
        retrieval_config.max_elements = ret_config.get("max_elements", 70000)
        retrieval_config.M = ret_config.get("M", 32)
        retrieval_config.ef_construction = ret_config.get("ef_construction", 200)
        retrieval_config.ef_search = ret_config.get("ef_search", 100)
        retrieval_config.random_seed = ret_config.get("random_seed", 42)
        retrieval_config.pool_size_bytes = ret_config.get("pool_size_bytes", 2_000_000_000)
    
    # If checkpoint provided, load model and generate embeddings
    if args.checkpoint and Path(args.checkpoint).exists():
        print("Loading checkpoint to extract item embeddings...")
        
        # Load checkpoint to get n_users and n_items
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Infer n_users from user_embedding weight shape
        user_emb_weight = state_dict.get("user_embedding.weight")
        if user_emb_weight is not None:
            n_users = user_emb_weight.shape[0]
        else:
            n_users = 162541  # ML-25M default
        
        # Infer n_items from item_embedding weight shape (checkpoint was trained on this many items)
        item_emb_weight = state_dict.get("item_embedding.weight")
        if item_emb_weight is not None:
            n_items = item_emb_weight.shape[0]
            print(f"Checkpoint n_items: {n_items}")
        else:
            n_items = item_metadata_np.shape[0]

        if n_items != item_metadata_np.shape[0]:
            raise ValueError(
                f"Checkpoint was trained on {n_items} items but movie_mapping.parquet has "
                f"{item_metadata_np.shape[0]}. The mapping changed since training — retrain "
                "or restore the matching mapping rather than truncating (that silently "
                "misaligns every index position)."
            )

        print(f"Inferred n_users={n_users}, n_items={n_items}, metadata_dim={metadata_dim}")

        ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model = TwoTowerWithMetadata(
            n_users=n_users,
            n_items=n_items,
            metadata_dim=metadata_dim,
            embedding_dim=ckpt_config.get("embedding_dim", 128),
            hidden_dim=ckpt_config.get("hidden_dim", 256),
            output_dim=ckpt_config.get("output_dim", 128),
            dropout=ckpt_config.get("dropout", 0.1),
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Warm path (item ID + metadata) — must match scripts/evaluate.py, otherwise we
        # measure one set of embeddings and serve another.
        item_metadata = torch.from_numpy(item_metadata_np).to(device)
        item_ids = torch.arange(n_items, dtype=torch.long, device=device)
        with torch.no_grad():
            item_embeddings = model.get_item_embeddings(item_ids, item_metadata)
            item_embeddings_np = item_embeddings.cpu().numpy().astype(np.float32)
    else:
        print("No checkpoint provided, using cold-start embeddings directly...")
        # The metadata IS already 128-dim projected embeddings
        item_embeddings_np = item_metadata_np.astype(np.float32)
    
    print(f"Item embeddings shape: {item_embeddings_np.shape}")
    
    # Build index
    index_path = Path(args.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    build_index(item_embeddings_np, index_path, retrieval_config)
    
    print(f"Index saved to {index_path}")
    print("Done!")


if __name__ == "__main__":
    main()