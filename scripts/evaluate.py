"""Offline evaluation script for Two-Tower movie recommender."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cold_start import build_aligned_metadata
from src.models.two_tower import TwoTowerWithMetadata, TwoTowerHistory


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_config = checkpoint.get("config", {})

    # Fallback: infer from state_dict if checkpoint config is incomplete (e.g., old best.pt)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    n_items = ckpt_config.get("n_items") or state_dict.get("item_embedding.weight", torch.empty(0)).shape[0]
    metadata_dim = ckpt_config.get("metadata_dim") or state_dict.get("item_metadata_proj.weight", torch.empty(0)).shape[1]
    embedding_dim = ckpt_config.get("embedding_dim", 128)
    hidden_dim = ckpt_config.get("hidden_dim", 256)
    output_dim = ckpt_config.get("output_dim", 128)
    dropout = ckpt_config.get("dropout", 0.1)

    if ckpt_config.get("model_type") == "two_tower_history" or "user_history" in state_dict:
        model = TwoTowerHistory(
            n_items=n_items,
            metadata_dim=metadata_dim,
            history=torch.zeros_like(state_dict["user_history"]),  # filled by load_state_dict
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    else:
        n_users = ckpt_config.get("n_users") or state_dict.get("user_embedding.weight", torch.empty(0)).shape[0]
        model = TwoTowerWithMetadata(
            n_users=n_users,
            n_items=n_items,
            metadata_dim=metadata_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

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


def get_user_embeddings(
    model,
    user_mapping: dict,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """Get embeddings for all users in mapping (chunked: the history model gathers
    a (B, K, dim) tensor per batch, which is too large for all users at once)."""
    user_ids = np.fromiter(user_mapping.values(), dtype=np.int64)
    out = np.zeros((len(user_ids), model.output_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(user_ids), batch_size):
            chunk = torch.from_numpy(user_ids[start : start + batch_size]).to(device)
            out[start : start + batch_size] = model.get_user_embeddings(chunk).cpu().numpy()

    return out


def compute_item_embeddings(
    model: TwoTowerWithMetadata,
    item_metadata: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute item embeddings via the warm path (item ID + metadata).

    item_metadata must be in movie_map index order (see build_aligned_metadata), which
    makes every row a trained item. Mirrors scripts/build_index.py so evaluation measures
    the same embeddings the API serves.
    """
    n_items = item_metadata.shape[0]
    print(f"  Computing embeddings for {n_items} items")

    all_embeddings = np.zeros((n_items, model.output_dim), dtype=np.float32)

    for i in range(0, n_items, batch_size):
        end = min(i + batch_size, n_items)
        batch_indices = np.arange(i, end)
        item_ids = torch.tensor(batch_indices, dtype=torch.long, device=device)
        metadata = torch.from_numpy(item_metadata[batch_indices]).to(device)

        with torch.no_grad():
            emb = model.get_item_embeddings(item_ids, metadata)

        all_embeddings[batch_indices] = emb.cpu().numpy().astype(np.float32)

    return all_embeddings


def load_seen_items(
    splits_dir: Path,
    user_mapping: dict,
    item_mapping: dict,
) -> tuple[dict, np.ndarray]:
    """Load each user's known interactions (everything except the held-out LOO item).

    Returns (user_idx -> np.ndarray of seen item_idx, popularity counts per item_idx).
    Seen items must be masked before top-K: recommending a movie the user already
    watched can never hit the held-out item, so leaving them in only deflates recall.
    """
    frames = [
        pd.read_parquet(splits_dir / f, columns=["userId", "movieId"])
        for f in ("train_loo.parquet", "val_loo.parquet")
    ]
    df = pd.concat(frames, ignore_index=True)

    u = df["userId"].map(user_mapping)
    i = df["movieId"].map(item_mapping)
    valid = u.notna() & i.notna()
    seen = pd.DataFrame({"u": u[valid].astype(np.int64), "i": i[valid].astype(np.int64)})

    pop_counts = np.zeros(len(item_mapping), dtype=np.float64)
    np.add.at(pop_counts, seen["i"].to_numpy(), 1)

    seen_dict = {int(uidx): grp.to_numpy() for uidx, grp in seen.groupby("u")["i"]}
    return seen_dict, pop_counts


def compute_metrics(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    seen_dict: dict,
    pop_counts: np.ndarray,
    user_mapping: dict,
    item_mapping: dict,
    k: int = 10,
    chunk_size: int = 512,
    popularity_weight: float = 0.0,
) -> dict:
    """Compute Recall@K (hit rate) and NDCG@K on the leave-one-out split.

    One held-out item per user, so NDCG@K reduces to 1/log2(rank+2) on a hit and
    recall to hit-or-miss. Also reports a popularity baseline (rank by train rating
    count, same seen-item masking) as a sanity floor for the model numbers.

    popularity_weight w ranks by `cosine + w*log(pop_count+1)` — the serving formula.
    The in-batch softmax loss learns popularity-corrected preferences, so raw cosine
    under-recommends popular movies; w adds that signal back. w=0 is pure cosine.
    """
    df = test_df[test_df["userId"].isin(user_mapping) & test_df["movieId"].isin(item_mapping)]
    print(f"Evaluating {len(df)} users out of {test_df['userId'].nunique()} in test set")

    users = df["userId"].map(user_mapping).to_numpy(np.int64)
    targets = df["movieId"].map(item_mapping).to_numpy(np.int64)

    pop_sorted = np.sort(pop_counts)
    n_items = len(pop_counts)
    pop_bonus = popularity_weight * np.log(pop_counts + 1.0).astype(np.float32)

    model_hits = model_ndcg = pop_hits = pop_ndcg = 0.0

    for start in range(0, len(users), chunk_size):
        u_chunk = users[start : start + chunk_size]
        t_chunk = targets[start : start + chunk_size]
        scores = user_embeddings[u_chunk] @ item_embeddings.T + pop_bonus

        for row in range(len(u_chunk)):
            uidx, target = int(u_chunk[row]), int(t_chunk[row])
            seen = seen_dict.get(uidx)

            s = scores[row]
            target_score = s[target]
            if seen is not None:
                s[seen] = -np.inf
            rank = int((s > target_score).sum())
            if rank < k:
                model_hits += 1
                model_ndcg += 1.0 / np.log2(rank + 2)

            # Popularity rank: count unseen items above the target; split ties evenly.
            pt = pop_counts[target]
            higher = n_items - int(np.searchsorted(pop_sorted, pt, side="right"))
            ties = int(np.searchsorted(pop_sorted, pt, side="right")) - int(
                np.searchsorted(pop_sorted, pt, side="left")
            ) - 1
            if seen is not None:
                seen_pop = pop_counts[seen]
                higher -= int((seen_pop > pt).sum())
                ties -= int((seen_pop == pt).sum())
            pop_rank = higher + ties // 2
            if pop_rank < k:
                pop_hits += 1
                pop_ndcg += 1.0 / np.log2(pop_rank + 2)

    n = len(users)
    return {
        f"recall@{k}": model_hits / n,
        f"ndcg@{k}": model_ndcg / n,
        f"popularity_recall@{k}": pop_hits / n,
        f"popularity_ndcg@{k}": pop_ndcg / n,
        "num_users_evaluated": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower on the leave-one-out split")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/two_tower/best_model.pt",
        help="Checkpoint to evaluate. Use an immutable epoch_N.pt when training is still "
             "running -- best_model.pt is rewritten on every val improvement and torch.save "
             "is not atomic.",
    )
    parser.add_argument("--output", type=str, default="outputs/eval_loo.json", help="Metrics output path")
    parser.add_argument("-k", type=int, default=10, help="Cutoff K for recall/NDCG")
    parser.add_argument(
        "--popularity-weight",
        type=float,
        default=None,
        help="Rank by cosine + w*log(pop+1); default comes from configs/retrieval.yaml",
    )
    args = parser.parse_args()

    popularity_weight = args.popularity_weight
    if popularity_weight is None:
        retrieval_cfg = load_config("configs/retrieval.yaml") if Path("configs/retrieval.yaml").exists() else {}
        popularity_weight = float(retrieval_cfg.get("popularity_weight", 0.0))
    print(f"Popularity weight: {popularity_weight}")

    data_config = load_config("configs/data.yaml")
    splits_dir = Path(data_config["splits"]["output_dir"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading mappings...")
    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    item_mapping = load_item_mapping(splits_dir / "movie_mapping.parquet")
    print(f"  Users: {len(user_mapping)}, Items: {len(item_mapping)}")

    print("Loading model...")
    model, ckpt_config = load_model(Path(args.checkpoint), device)
    ckpt_n_items = ckpt_config["n_items"]
    print(f"  Checkpoint n_items: {ckpt_n_items}")

    print("Computing user embeddings...")
    user_embeddings = get_user_embeddings(model, user_mapping, device)
    print(f"  User embeddings shape: {user_embeddings.shape}")

    print("Loading item metadata...")
    item_metadata = build_aligned_metadata(item_mapping, splits_dir)

    print("Computing item embeddings...")
    item_embeddings = compute_item_embeddings(model, item_metadata, device)
    print(f"  Item embeddings shape: {item_embeddings.shape}")

    print("Loading test data...")
    test_df = pd.read_parquet(splits_dir / "test_loo.parquet")
    print(f"  Test interactions: {len(test_df)}")

    print("Loading seen items (for masking) and popularity counts...")
    seen_dict, pop_counts = load_seen_items(splits_dir, user_mapping, item_mapping)

    # Serving artifact: infer.py / app load these counts to apply the same
    # popularity blend at retrieval time (movie_map index order).
    np.save(splits_dir / "popularity_counts.npy", pop_counts)

    print("Computing metrics...")
    metrics = compute_metrics(
        user_embeddings,
        item_embeddings,
        test_df,
        seen_dict,
        pop_counts,
        user_mapping,
        item_mapping,
        k=args.k,
        popularity_weight=popularity_weight,
    )
    metrics["popularity_weight"] = popularity_weight

    print("\n=== Evaluation Results ===")
    for key, v in metrics.items():
        print(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"protocol": "leave_one_out", "checkpoint": args.checkpoint, "metrics": metrics},
            f,
            indent=2,
        )
    print(f"\nWrote metrics to {output_path}")


if __name__ == "__main__":
    main()