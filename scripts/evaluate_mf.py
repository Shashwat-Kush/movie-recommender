"""Evaluate an ImplicitMF (BPR) checkpoint on the leave-one-out split.

Same protocol as scripts/evaluate.py (seen-item masking, popularity baseline) so the
numbers are directly comparable with the Two-Tower results. BPR scores are raw dot
products of the factor tables — BPR learns popularity natively, so no blend.

Usage: PYTHONPATH=. python3 scripts/evaluate_mf.py --checkpoint checkpoints/mf_implicit_loo/epoch_2.pt
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
    load_config, load_user_mapping, load_item_mapping, load_seen_items, compute_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mf_implicit_loo/epoch_2.pt")
    parser.add_argument("--output", type=str, default="outputs/eval_loo_mf.json")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    data_config = load_config("configs/data.yaml")
    splits_dir = Path(data_config["splits"]["output_dir"])

    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    item_mapping = load_item_mapping(splits_dir / "movie_mapping.parquet")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    user_embeddings = state_dict["user_embeddings.weight"].numpy().astype(np.float32)
    item_embeddings = state_dict["item_embeddings.weight"].numpy().astype(np.float32)
    print(f"Factors: users {user_embeddings.shape}, items {item_embeddings.shape}")

    test_df = pd.read_parquet(splits_dir / "test_loo.parquet")
    seen_dict, pop_counts = load_seen_items(splits_dir, user_mapping, item_mapping)

    metrics = compute_metrics(
        user_embeddings, item_embeddings, test_df, seen_dict, pop_counts,
        user_mapping, item_mapping, k=args.k, popularity_weight=0.0,
    )

    print("\n=== ImplicitMF (BPR) Evaluation ===")
    for key, v in metrics.items():
        print(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": "leave_one_out", "checkpoint": args.checkpoint, "metrics": metrics}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
