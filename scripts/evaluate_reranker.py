"""Offline A/B evaluation of the Groq reranker (retrieval order vs reranked order).

For a sample of LOO users: take the blended, seen-filtered top-50 retrieval
candidates, build a taste-profile query from the user's most recent liked movies,
let the reranker pick its top-10, and check whether the held-out item ranks higher
than in plain retrieval order. Reports Recall@10 / NDCG@10 for both arms.

Usage: PYTHONPATH=. python3 scripts/evaluate_reranker.py [--users 200] [--cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from scripts.evaluate import (
    load_config, load_model, load_user_mapping, load_item_mapping,
    get_user_embeddings, compute_item_embeddings, load_seen_items,
)
from src.data.cold_start import build_aligned_metadata
from src.models.reranker import create_reranker

K = 10
CANDIDATES = 50
HISTORY_TITLES = 5


def ndcg_single(rank: int) -> float:
    return 1.0 / np.log2(rank + 2) if rank < K else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/two_tower/best_model.pt")
    parser.add_argument("--users", type=int, default=200, help="Users to sample")
    parser.add_argument("--output", type=str, default="outputs/eval_reranker_ab.json")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (leave MPS free for training)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.cpu:
        torch.backends.mps.is_available = lambda: False
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data_config = load_config("configs/data.yaml")
    splits_dir = Path(data_config["splits"]["output_dir"])
    retrieval_cfg = load_config("configs/retrieval.yaml")
    pop_weight = float(retrieval_cfg.get("popularity_weight", 0.0))

    user_mapping = load_user_mapping(splits_dir / "user_mapping.parquet")
    item_mapping = load_item_mapping(splits_dir / "movie_mapping.parquet")
    idx_to_movie_id = {v: k for k, v in item_mapping.items()}

    model, _ = load_model(Path(args.checkpoint), device)
    user_embeddings = get_user_embeddings(model, user_mapping, device)
    item_metadata = build_aligned_metadata(item_mapping, splits_dir)
    item_embeddings = compute_item_embeddings(model, item_metadata, device)

    seen_dict, pop_counts = load_seen_items(splits_dir, user_mapping, item_mapping)
    pop_bonus = pop_weight * np.log(pop_counts + 1.0).astype(np.float32)

    # Movie titles/genres for prompts (small table, load fully)
    movies = ds.dataset("data/parquet/movies").to_table(columns=["movieId", "title", "genres"]).to_pandas()
    movies = movies.set_index("movieId")

    # Sample users from the LOO test set
    test_df = pd.read_parquet(splits_dir / "test_loo.parquet")
    test_df = test_df[test_df["userId"].isin(user_mapping) & test_df["movieId"].isin(item_mapping)]
    rng = np.random.default_rng(args.seed)
    sample = test_df.iloc[rng.choice(len(test_df), size=args.users, replace=False)]

    # Recent liked movies per sampled user (for the taste-profile prompt)
    hist = ds.dataset(splits_dir / "train_loo.parquet").to_table(
        columns=["userId", "movieId", "rating", "timestamp"],
        filter=ds.field("userId").isin(sample["userId"].tolist()),
    ).to_pandas()
    hist = hist[hist["rating"] >= 4.0].sort_values("timestamp", ascending=False)
    hist_by_user = {u: g["movieId"].head(HISTORY_TITLES).tolist() for u, g in hist.groupby("userId")}

    reranker = create_reranker(load_config("configs/reranker.yaml") if Path("configs/reranker.yaml").exists() else {})

    ret_hits = ret_ndcg = rr_hits = rr_ndcg = 0.0
    evaluated = fallbacks = 0

    for _, row in sample.iterrows():
        user_id, target_movie = int(row["userId"]), int(row["movieId"])
        uidx, target_idx = user_mapping[user_id], item_mapping[target_movie]

        scores = item_embeddings @ user_embeddings[uidx] + pop_bonus
        seen = seen_dict.get(uidx)
        if seen is not None:
            scores[seen] = -np.inf
        top = np.argpartition(-scores, CANDIDATES)[:CANDIDATES]
        top = top[np.argsort(-scores[top])]

        # Retrieval arm
        ret_rank = np.where(top[:K] == target_idx)[0]
        ret_rank = int(ret_rank[0]) if len(ret_rank) else K
        ret_hits += ret_rank < K
        ret_ndcg += ndcg_single(ret_rank)

        # Reranker arm: taste-profile query over the same 50 candidates
        liked = [str(movies.loc[m, "title"]) for m in hist_by_user.get(user_id, []) if m in movies.index]
        query = "Movies matching this user's taste. They recently loved: " + "; ".join(liked) if liked else "Widely appealing movies"
        candidates = []
        for idx in top:
            mid = idx_to_movie_id[int(idx)]
            title = str(movies.loc[mid, "title"]) if mid in movies.index else ""
            genres = str(movies.loc[mid, "genres"]) if mid in movies.index else ""
            candidates.append({"movieId": mid, "title": title, "genres": genres, "tags": ""})

        try:
            reranked = reranker.rerank(query, candidates, top_k=K)
        except Exception as e:
            print(f"  user {user_id}: reranker error ({e}), retrying in 20s")
            time.sleep(20)
            try:
                reranked = reranker.rerank(query, candidates, top_k=K)
            except Exception:
                continue

        if reranked and all(m.get("rerank_score", 0) == 0.0 for m in reranked):
            fallbacks += 1
        rr_ids = [m["movieId"] for m in reranked]
        rr_rank = rr_ids.index(target_movie) if target_movie in rr_ids else K
        rr_hits += rr_rank < K
        rr_ndcg += ndcg_single(rr_rank)

        evaluated += 1
        if evaluated % 20 == 0:
            print(f"  [{evaluated}/{args.users}] retrieval recall {ret_hits/evaluated:.3f} | reranked {rr_hits/evaluated:.3f}")

    results = {
        "users_evaluated": evaluated,
        "reranker_fallbacks": fallbacks,
        "retrieval": {f"recall@{K}": ret_hits / evaluated, f"ndcg@{K}": ret_ndcg / evaluated},
        "reranked": {f"recall@{K}": rr_hits / evaluated, f"ndcg@{K}": rr_ndcg / evaluated},
    }
    print("\n=== Reranker A/B Results ===")
    print(json.dumps(results, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
