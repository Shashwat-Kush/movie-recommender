"""Cold-start measurement via a pseudo-cold protocol.

Samples a fraction of catalog items and embeds them through the metadata-only cold
path (zero ID embedding + MiniLM metadata, model.get_item_embeddings_cold) while the
rest stay warm. Users whose held-out LOO item falls in the pseudo-cold set are then
evaluated twice on identical candidates: once with the cold catalog and once fully
warm. The paired gap isolates what the ID embedding contributes — i.e., how much
recall a genuinely new movie would lose.

Usage: PYTHONPATH=. python3 scripts/evaluate_cold_start.py [--cold-frac 0.2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import (
    load_config, load_model, load_user_mapping, load_item_mapping,
    get_user_embeddings, compute_item_embeddings, load_seen_items, compute_metrics,
)
from src.data.cold_start import build_aligned_metadata


def compute_cold_embeddings(model, item_metadata: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    out = np.zeros((item_metadata.shape[0], model.output_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, item_metadata.shape[0], batch_size):
            meta = torch.from_numpy(item_metadata[start : start + batch_size]).to(device)
            out[start : start + batch_size] = model.get_item_embeddings_cold(meta).cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/two_tower_history/best_model.pt")
    parser.add_argument("--cold-frac", type=float, default=0.2, help="Fraction of items made pseudo-cold")
    parser.add_argument("--output", type=str, default="outputs/eval_cold_start.json")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_config = load_config("configs/data.yaml")
    splits_dir = Path(data_config["splits"]["output_dir"])
    retrieval_cfg = load_config("configs/retrieval.yaml") if Path("configs/retrieval.yaml").exists() else {}
    popularity_weight = float(retrieval_cfg.get("popularity_weight", 0.0))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    item_mapping = load_item_mapping(splits_dir / "movie_mapping.parquet")

    model, _ = load_model(Path(args.checkpoint), device)
    user_embeddings = get_user_embeddings(model, user_mapping, device)
    item_metadata = build_aligned_metadata(item_mapping, splits_dir)
    warm_embeddings = compute_item_embeddings(model, item_metadata, device)
    cold_embeddings = compute_cold_embeddings(model, item_metadata, device)

    n_items = warm_embeddings.shape[0]
    rng = np.random.default_rng(args.seed)
    cold_items = rng.random(n_items) < args.cold_frac
    print(f"Pseudo-cold items: {int(cold_items.sum()):,}/{n_items:,}")

    # Catalog with the sampled items served cold, the rest warm
    mixed_embeddings = warm_embeddings.copy()
    mixed_embeddings[cold_items] = cold_embeddings[cold_items]

    test_df = pd.read_parquet(splits_dir / "test_loo.parquet")
    test_df = test_df[test_df["userId"].isin(user_mapping) & test_df["movieId"].isin(item_mapping)]
    target_idx = test_df["movieId"].map(item_mapping).to_numpy()
    cold_targets = test_df[cold_items[target_idx]]
    print(f"Users with a pseudo-cold held-out item: {len(cold_targets):,}")

    seen_dict, pop_counts = load_seen_items(splits_dir, user_mapping, item_mapping)

    results = {"checkpoint": args.checkpoint, "cold_frac": args.cold_frac, "popularity_weight": popularity_weight}
    for arm, embeddings in (("cold", mixed_embeddings), ("warm", warm_embeddings)):
        print(f"\n--- {arm} catalog, same {len(cold_targets):,} users ---")
        metrics = compute_metrics(
            user_embeddings, embeddings, cold_targets, seen_dict, pop_counts,
            user_mapping, item_mapping, k=args.k, popularity_weight=popularity_weight,
        )
        for key, v in metrics.items():
            print(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")
        results[arm] = metrics

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
