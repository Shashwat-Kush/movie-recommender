"""Temporal robustness evaluation: LOO recall bucketed by held-out interaction period.

Buckets each user's held-out item by the time-split windows from configs/data.yaml
(train / val / test / post-test) and reports Recall@K / NDCG@K per bucket. This
measures how ranking quality varies with the recency of the target interaction.

Caveat: the served model trains on every user's history regardless of period (LOO
protocol), so this is NOT a strict "train on past, predict future" evaluation —
that would require retraining on the time-split train set only.

Usage: PYTHONPATH=. python3 scripts/evaluate_timesplit.py
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import (
    load_config, load_model, load_user_mapping, load_item_mapping,
    get_user_embeddings, compute_item_embeddings, load_seen_items, compute_metrics,
)
from src.data.cold_start import build_aligned_metadata


def ts(date_str: str) -> int:
    return int(datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/two_tower_history/best_model.pt")
    parser.add_argument("--output", type=str, default="outputs/eval_timesplit.json")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--popularity-weight", type=float, default=None)
    args = parser.parse_args()

    data_config = load_config("configs/data.yaml")
    splits_dir = Path(data_config["splits"]["output_dir"])
    cuts = data_config["splits"]["time_split"]

    popularity_weight = args.popularity_weight
    if popularity_weight is None:
        retrieval_cfg = load_config("configs/retrieval.yaml") if Path("configs/retrieval.yaml").exists() else {}
        popularity_weight = float(retrieval_cfg.get("popularity_weight", 0.0))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    item_mapping = load_item_mapping(splits_dir / "movie_mapping.parquet")

    model, _ = load_model(Path(args.checkpoint), device)
    user_embeddings = get_user_embeddings(model, user_mapping, device)
    item_metadata = build_aligned_metadata(item_mapping, splits_dir)
    item_embeddings = compute_item_embeddings(model, item_metadata, device)

    test_df = pd.read_parquet(splits_dir / "test_loo.parquet")
    seen_dict, pop_counts = load_seen_items(splits_dir, user_mapping, item_mapping)

    buckets = {
        f"train_window (<{cuts['train_cutoff']})": test_df["timestamp"] < ts(cuts["train_cutoff"]),
        f"val_window ({cuts['train_cutoff']}..{cuts['val_cutoff']})": (
            (test_df["timestamp"] >= ts(cuts["train_cutoff"])) & (test_df["timestamp"] < ts(cuts["val_cutoff"]))
        ),
        f"test_window ({cuts['val_cutoff']}..{cuts['test_cutoff']})": (
            (test_df["timestamp"] >= ts(cuts["val_cutoff"])) & (test_df["timestamp"] < ts(cuts["test_cutoff"]))
        ),
        f"post_test (>={cuts['test_cutoff']})": test_df["timestamp"] >= ts(cuts["test_cutoff"]),
    }

    results = {"checkpoint": args.checkpoint, "popularity_weight": popularity_weight, "buckets": {}}
    for name, mask in buckets.items():
        subset = test_df[mask]
        if subset.empty:
            continue
        print(f"\n--- {name}: {len(subset):,} held-out interactions ---")
        metrics = compute_metrics(
            user_embeddings, item_embeddings, subset, seen_dict, pop_counts,
            user_mapping, item_mapping, k=args.k, popularity_weight=popularity_weight,
        )
        for key, v in metrics.items():
            print(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")
        results["buckets"][name] = metrics

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
