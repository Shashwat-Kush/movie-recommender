"""Capture real API responses and offline artifacts as frontend fixtures.

Everything written here is genuinely produced by the serving stack (via FastAPI
TestClient, which runs the real startup + endpoints) or derived offline from real
data — the static frontend replays these, it never invents numbers.

Outputs (frontend/fixtures/):
  recommendations.json  — recorded /recommend and /recommend_cold responses, keyed
  users.json            — demo users with genre/decade taste profiles from their history
  movies.json           — popular-catalog slice for the cold-start picker
  projection.json       — 2D PCA of the served model's item embeddings
  eval.json             — v2 evaluation metrics + static system facts

Usage: PYTHONPATH=. python3 scripts/capture_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

FIXTURES = Path("frontend/fixtures")
DEMO_USERS = [1, 42, 7, 33844, 96421]
QUERIES = [
    "a mind-bending sci-fi thriller",
    "feel-good comedy for a rainy day",
    "dark crime drama with a great ending",
]
COLD_SCENARIOS = [
    {
        "name": "animation-fan",
        "liked_movie_ids": [1, 4886, 6377, 68954, 78499],  # Toy Story, Monsters Inc, Nemo, Up, TS3
        "query": "something the whole family will love",
    },
    {
        "name": "thriller-fan",
        "liked_movie_ids": [2571, 48780, 4226, 74458, 1682],  # Matrix, Prestige, Memento, Shutter Island, Truman Show
        "query": "a smart thriller that keeps me guessing",
    },
]
TOP_K = 10


def year_of(title: str):
    import re
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)

    movies = pd.read_parquet("data/parquet/movies")[["movieId", "title", "genres"]]
    train = pd.read_parquet(
        "data/processed/train_loo.parquet", columns=["userId", "movieId", "rating", "timestamp"]
    )

    # --- users.json: taste profiles from real rating history -------------------
    users_out = []
    titles_by_id = movies.set_index("movieId")
    for uid in DEMO_USERS:
        hist = train[train["userId"] == uid].merge(movies, on="movieId")
        liked = hist[hist["rating"] >= 3.5]
        genre_counts = liked["genres"].str.split("|").explode().value_counts()
        genre_dist = (genre_counts / genre_counts.sum()).round(4).head(8).to_dict()
        years = liked["title"].map(year_of).dropna()
        decade_counts = (years // 10 * 10).astype(int).value_counts().sort_index()
        decade_dist = {str(k): round(v / len(years), 4) for k, v in decade_counts.items()}
        recent = liked.sort_values("timestamp", ascending=False).head(5)["title"].tolist()
        users_out.append({
            "user_id": uid,
            "num_ratings": int(len(hist)),
            "num_liked": int(len(liked)),
            "genre_distribution": genre_dist,
            "decade_distribution": decade_dist,
            "recent_liked_titles": recent,
        })
    (FIXTURES / "users.json").write_text(json.dumps(users_out, indent=2))
    print(f"users.json: {len(users_out)} demo users")

    # --- movies.json: popular slice for the cold-start picker ------------------
    pop = train[train["rating"] >= 3.5]["movieId"].value_counts().head(250)
    picker = movies[movies["movieId"].isin(pop.index)].copy()
    picker["popularity"] = picker["movieId"].map(pop)
    picker["year"] = picker["title"].map(year_of)
    picker = picker.sort_values("popularity", ascending=False)
    (FIXTURES / "movies.json").write_text(
        picker[["movieId", "title", "genres", "year"]].to_json(orient="records")
    )
    print(f"movies.json: {len(picker)} picker movies")

    # --- projection.json: 2D PCA of served item embeddings ---------------------
    from scripts.evaluate import load_config, load_model, load_item_mapping, compute_item_embeddings
    from src.data.cold_start import build_aligned_metadata

    device = torch.device("cpu")
    item_mapping = load_item_mapping(Path("data/processed/movie_mapping.parquet"))
    model, _ = load_model(Path("checkpoints/two_tower_history_v2/best_model.pt"), device)
    metadata = build_aligned_metadata(item_mapping, Path("data/processed"))
    emb = compute_item_embeddings(model, metadata, device)

    centered = emb - emb.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T

    # Keep the ~3000 most popular items for a legible scatter
    inv = {v: k for k, v in item_mapping.items()}
    all_pop = train[train["rating"] >= 3.5]["movieId"].value_counts()
    idx_pop = np.array([all_pop.get(inv[i], 0) for i in range(len(inv))])
    keep = np.argsort(-idx_pop)[:3000]
    points = []
    for i in keep:
        mid = inv[int(i)]
        if mid not in titles_by_id.index:
            continue
        row = titles_by_id.loc[mid]
        points.append({
            "movieId": int(mid),
            "title": str(row["title"]),
            "genre": str(row["genres"]).split("|")[0],
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
        })
    (FIXTURES / "projection.json").write_text(json.dumps(points))
    print(f"projection.json: {len(points)} points")

    # --- eval.json: v2 metrics + static system facts ----------------------------
    eval_v2 = json.loads(Path("outputs/eval_loo_v2.json").read_text())
    cold = json.loads(Path("outputs/eval_cold_start_v2.json").read_text())
    timesplit = json.loads(Path("outputs/eval_timesplit_v2.json").read_text())
    (FIXTURES / "eval.json").write_text(json.dumps({
        "loo": eval_v2["metrics"],
        "cold_start": {"cold": cold["cold"], "warm": cold["warm"], "cold_frac": cold["cold_frac"]},
        "timesplit": timesplit["buckets"],
        "system": {
            "movies_indexed": 26744,
            "users_trained": 138493,
            "ratings_trained": "25M MovieLens ratings",
            "embedding_dim": 128,
            "hnsw": {"M": 32, "ef_construction": 200, "ef_search": 100},
            "reranker_model": "gpt-oss-120b",
            "candidate_pool": 500,
            "rerank_candidates": 25,
        },
    }, indent=2))
    print("eval.json written")

    # --- recommendations.json: real recorded API responses ---------------------
    from app.main import app

    fixtures = {"recommend": [], "recommend_cold": []}
    with TestClient(app) as client:
        for uid in DEMO_USERS:
            for query in QUERIES:
                r = client.post("/recommend", json={"user_id": uid, "query": query, "top_k": TOP_K})
                r.raise_for_status()
                fixtures["recommend"].append({
                    "request": {"user_id": uid, "query": query, "top_k": TOP_K},
                    "response": r.json(),
                })
                print(f"  /recommend user={uid} '{query}' ok")
        for scenario in COLD_SCENARIOS:
            body = {k: scenario[k] for k in ("liked_movie_ids", "query")} | {"top_k": TOP_K}
            r = client.post("/recommend_cold", json=body)
            r.raise_for_status()
            fixtures["recommend_cold"].append({
                "name": scenario["name"], "request": body, "response": r.json(),
            })
            print(f"  /recommend_cold '{scenario['name']}' ok")

    (FIXTURES / "recommendations.json").write_text(json.dumps(fixtures, indent=2))
    n = len(fixtures["recommend"]) + len(fixtures["recommend_cold"])
    print(f"recommendations.json: {n} recorded responses")


if __name__ == "__main__":
    main()
