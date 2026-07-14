"""Offline evaluation script for Two-Tower movie recommender."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerWithMetadata


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TwoTowerWithMetadata, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_config = checkpoint.get("config", {})

    # Fallback: infer from state_dict if checkpoint config is incomplete (e.g., old best.pt)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    n_users = ckpt_config.get("n_users") or state_dict.get("user_embedding.weight", torch.empty(0)).shape[0]
    n_items = ckpt_config.get("n_items") or state_dict.get("item_embedding.weight", torch.empty(0)).shape[0]
    metadata_dim = ckpt_config.get("metadata_dim") or state_dict.get("item_metadata_proj.weight", torch.empty(0)).shape[1]
    embedding_dim = ckpt_config.get("embedding_dim", 128)
    hidden_dim = ckpt_config.get("hidden_dim", 256)
    output_dim = ckpt_config.get("output_dim", 128)
    dropout = ckpt_config.get("dropout", 0.1)

    model = TwoTowerWithMetadata(
        n_users=n_users,
        n_items=n_items,
        metadata_dim=metadata_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, ckpt_config


def load_user_mapping(mapping_path: Path) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(mapping_path)
    return dict(zip(table.column("userId").to_pylist(), table.column("user_idx").to_pylist()))


def load_item_mapping(mapping_path: Path) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(mapping_path)
    return dict(zip(table.column("movieId").to_pylist(), table.column("movie_idx").to_pylist()))


def get_user_embeddings(model: TwoTowerWithMetadata, user_mapping: dict, device: torch.device) -> np.ndarray:
    """Get embeddings for all users in mapping."""
    user_ids = list(user_mapping.values())
    user_tensor = torch.tensor(user_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        embeddings = model.get_user_embeddings(user_tensor)

    return embeddings.cpu().numpy().astype(np.float32)


def compute_item_embeddings(
    model: TwoTowerWithMetadata,
    item_metadata: np.ndarray,
    ckpt_n_items: int,
    full_n_items: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute item embeddings using trained model.

    For items in checkpoint (0 to ckpt_n_items-1): use trained item_id embedding + metadata.
    For remaining items: use cold-start (zero embedding + metadata).
    """
    print(f"  Computing embeddings for {full_n_items} items (trained: {ckpt_n_items}, cold-start: {full_n_items - ckpt_n_items})")

    # Get trained item ID embeddings
    trained_item_emb = model.item_embedding.weight.data.cpu().numpy()  # (ckpt_n_items, 128)

    all_embeddings = np.zeros((full_n_items, model.output_dim), dtype=np.float32)

    # Process trained items
    for i in range(0, ckpt_n_items, batch_size):
        end = min(i + batch_size, ckpt_n_items)
        batch_indices = np.arange(i, end)
        item_ids = torch.tensor(batch_indices, dtype=torch.long, device=device)
        metadata = torch.from_numpy(item_metadata[batch_indices]).to(device)

        with torch.no_grad():
            emb = model.get_item_embeddings(item_ids, metadata)

        all_embeddings[batch_indices] = emb.cpu().numpy().astype(np.float32)

    # Process cold-start items
    for i in range(ckpt_n_items, full_n_items, batch_size):
        end = min(i + batch_size, full_n_items)
        batch_indices = np.arange(i, end)
        metadata = torch.from_numpy(item_metadata[batch_indices]).to(device)

        with torch.no_grad():
            emb = model.get_item_embeddings_cold(metadata)

        all_embeddings[batch_indices] = emb.cpu().numpy().astype(np.float32)

    return all_embeddings


def compute_metrics(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    user_mapping: dict,
    item_mapping: dict,
    k: int = 10,
) -> dict:
    """Compute Recall@K and NDCG@K."""

    test_users = test_df["userId"].unique()
    valid_users = [u for u in test_users if u in user_mapping]

    print(f"Evaluating {len(valid_users)} users out of {len(test_users)} in test set")

    all_recall = []
    all_ndcg = []

    user_emb_lookup = {u: user_embeddings[user_mapping[u]] for u in valid_users}
    # Map movieId -> item_idx for fast lookup
    item_idx_lookup = {movie_id: item_mapping[movie_id] for movie_id in item_mapping}

    for user_id in valid_users:
        user_emb = user_emb_lookup[user_id]

        user_test = test_df[test_df["userId"] == user_id]
        true_items = user_test["movieId"].values
        true_ratings = user_test["rating"].values

        valid_item_indices = [item_idx_lookup[i] for i in true_items if i in item_idx_lookup]
        if not valid_item_indices:
            continue

        scores = item_embeddings @ user_emb
        rated_item_indices = np.array(valid_item_indices)
        rated_scores = scores[rated_item_indices]

        top_k_indices = np.argpartition(-scores, k)[:k]
        top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]
        top_k_set = set(top_k_indices)

        hits = sum(1 for idx in rated_item_indices if idx in top_k_set)
        recall = hits / len(rated_item_indices)
        all_recall.append(recall)

        dcg = 0.0
        for rank, idx in enumerate(top_k_indices):
            if idx in rated_item_indices:
                rel = true_ratings[np.where(rated_item_indices == idx)[0][0]]
                dcg += (2 ** rel - 1) / np.log2(rank + 2)

        ideal_ratings = np.sort(true_ratings)[::-1][:k]
        idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_ratings))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        all_ndcg.append(ndcg)

    return {
        f"recall@{k}": float(np.mean(all_recall)),
        f"ndcg@{k}": float(np.mean(all_ndcg)),
        "num_users_evaluated": len(all_recall),
    }


def main():
    data_config = load_config("configs/data.yaml")
    splits_dir = Path(data_config["splits"]["output_dir"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading mappings...")
    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    item_mapping = load_item_mapping(splits_dir / "movie_mapping.parquet")
    print(f"  Users: {len(user_mapping)}, Items: {len(item_mapping)}")

    print("Loading model...")
    model, ckpt_config = load_model(
        Path("checkpoints/two_tower/best_model.pt"),
        device,
    )
    ckpt_n_items = ckpt_config["n_items"]
    print(f"  Checkpoint n_items: {ckpt_n_items}")

    print("Computing user embeddings...")
    user_embeddings = get_user_embeddings(model, user_mapping, device)
    print(f"  User embeddings shape: {user_embeddings.shape}")

    print("Loading item metadata...")
    item_metadata = np.load("data/processed/cold_start_embeddings_128.npy").astype(np.float32)
    print(f"  Item metadata shape: {item_metadata.shape}")
    full_n_items = len(item_metadata)

    print("Computing item embeddings...")
    item_embeddings = compute_item_embeddings(
        model, item_metadata, ckpt_n_items, full_n_items, device
    )
    print(f"  Item embeddings shape: {item_embeddings.shape}")

    print("Loading test data...")
    test_df = pd.read_parquet(splits_dir / "test_loo.parquet")
    print(f"  Test interactions: {len(test_df)}")

    print("Computing metrics...")
    metrics = compute_metrics(
        user_embeddings,
        item_embeddings,
        test_df,
        user_mapping,
        item_mapping,
        k=10,
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()