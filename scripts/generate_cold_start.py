#!/usr/bin/env python
"""Stage 3: Offline batch encoding of cold-start metadata using MiniLM-L6-v2.

Generates 384-dim embeddings and learns a 384->128 projection layer
trained on items with both CF and text embeddings (MSE loss).
"""

import gc
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from src.data.cold_start import load_cold_start_data, load_corpus, load_config
from src.data.dataset import build_id_mappings


def encode_corpus(model: SentenceTransformer, corpus: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode text corpus in batches to 384-dim embeddings."""
    all_embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i : i + batch_size]
        with torch.inference_mode():
            emb = model.encode(batch, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(emb)
        del emb
        gc.collect()
    return np.vstack(all_embeddings).astype(np.float32)


def train_projection(
    text_emb: np.ndarray,
    cf_emb: np.ndarray,
    movie_ids: list[int],
    cf_movie_ids: list[int],
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "mps",
) -> nn.Linear:
    """Learn Linear(384, 128) projection via MSE on overlapping items."""
    # Align by movieId
    cf_id_to_idx = {mid: i for i, mid in enumerate(cf_movie_ids)}
    overlap_idx_text = []
    overlap_idx_cf = []
    for i, mid in enumerate(movie_ids):
        if mid in cf_id_to_idx:
            overlap_idx_text.append(i)
            overlap_idx_cf.append(cf_id_to_idx[mid])

    if not overlap_idx_text:
        raise ValueError("No overlapping movie IDs between text corpus and CF embeddings")

    X = torch.from_numpy(text_emb[overlap_idx_text]).to(device)
    y = torch.from_numpy(cf_emb[overlap_idx_cf]).to(device)

    proj = nn.Linear(384, 128, bias=False).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    proj.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = proj(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"  Projection epoch {epoch}: MSE = {loss.item():.6f}")

    proj.eval()
    with torch.inference_mode():
        final_loss = loss_fn(proj(X), y).item()
    print(f"  Final projection MSE: {final_loss:.6f}")
    return proj


def main():
    parser = argparse.ArgumentParser(description="Generate cold-start embeddings")
    parser.add_argument("--config", default="configs/data.yaml", help="Data config path")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Transformer model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--mf-checkpoint", default="checkpoints/matrix_factorization/best.pt", help="MF checkpoint for projection training")
    parser.add_argument("--epochs", type=int, default=50, help="Projection training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Projection learning rate")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", help="Device for projection training")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config["splits"]["output_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = processed_dir / "cold_start_corpus.parquet"
    if corpus_path.exists():
        print(f"Loading corpus from {corpus_path}")
        movie_ids, corpus = load_corpus(corpus_path)
    else:
        print("Building corpus from parquet...")
        movie_ids, corpus = load_cold_start_data(config)
        from src.data.cold_start import save_corpus
        save_corpus(movie_ids, corpus, corpus_path)

    print(f"Corpus size: {len(corpus):,} movies")

    # Load sentence transformer
    print(f"Loading model: {args.model}")
    device = "mps" if torch.backends.mps.is_available() and args.device == "mps" else "cpu"
    model = SentenceTransformer(args.model, device=device)

    # Encode corpus
    print(f"Encoding corpus (batch_size={args.batch_size})...")
    text_embeddings = encode_corpus(model, corpus, batch_size=args.batch_size)
    print(f"  Embeddings shape: {text_embeddings.shape}")

    # Save movieId alignment array (same order as embeddings) for downstream consumers
    movie_ids_path = processed_dir / "cold_start_movie_ids.npy"
    np.save(movie_ids_path, np.array(movie_ids, dtype=np.int32))
    print(f"Saved movieId alignment array to {movie_ids_path}")

    # Save 384-dim embeddings
    embeddings_path = processed_dir / "cold_start_embeddings.npy"
    np.save(embeddings_path, text_embeddings)
    print(f"Saved 384-dim embeddings to {embeddings_path}")

    # Load MF item embeddings for projection training
    mf_ckpt = Path(args.mf_checkpoint)
    if mf_ckpt.exists():
        print(f"Loading MF embeddings from {mf_ckpt}")
        mf_checkpoint = torch.load(mf_ckpt, map_location="cpu")
        state_dict = mf_checkpoint.get("model_state_dict", mf_checkpoint)
        for key in ["item_embeddings.weight", "item_emb.weight", "item_embeddings", "item_factors", "item_emb"]:
            if key in state_dict:
                emb = state_dict[key]
                if isinstance(emb, torch.Tensor):
                    cf_emb = emb.detach().cpu().numpy()
                    break
        else:
            for key, val in state_dict.items():
                if isinstance(val, torch.Tensor) and val.ndim == 2:
                    cf_emb = val.detach().cpu().numpy()
                    break
            else:
                cf_emb = None

        if cf_emb is not None:
            n_items_cf, emb_dim = cf_emb.shape
            print(f"  CF embeddings: {cf_emb.shape}")

            # Load movie ID mapping saved during MF training (matches CF embedding indices)
            movie_mapping_path = processed_dir / "movie_mapping.parquet"
            if movie_mapping_path.exists():
                print(f"Loading movie ID mapping from {movie_mapping_path}")
                movie_map_table = pq.read_table(movie_mapping_path)
                movie_id_map = dict(zip(movie_map_table["movieId"].to_pylist(), movie_map_table["movie_idx"].to_pylist()))
            else:
                # Fallback: build from full ratings data (may miss cold-start items)
                print("Mapping file not found, building from full ratings data...")
                parquet_root = Path(config["parquet"]["output_dir"])
                ratings_parquet = parquet_root / "ratings"
                _, movie_id_map = build_id_mappings(ratings_parquet)
            
            # Reverse map: index -> raw movieId (CF embedding index -> movieId)
            idx_to_movie = {idx: mid for mid, idx in movie_id_map.items()}
            cf_movie_ids = [idx_to_movie[i] for i in range(n_items_cf)]
            print(f"  CF movie IDs (mapped): {len(cf_movie_ids)}")

            # Train projection
            print("Training 384->128 projection layer...")
            proj = train_projection(
                text_embeddings, cf_emb, movie_ids, cf_movie_ids,
                epochs=args.epochs, lr=args.lr, device=args.device
            )

            # Save projection weights
            proj_path = processed_dir / "cold_start_projection.pt"
            torch.save(proj.state_dict(), proj_path)
            print(f"Saved projection to {proj_path}")

            # Save projected 128-dim embeddings
            with torch.inference_mode():
                projected = proj(torch.from_numpy(text_embeddings).to(args.device)).cpu().numpy()
            proj_emb_path = processed_dir / "cold_start_embeddings_128.npy"
            np.save(proj_emb_path, projected.astype(np.float32))
            print(f"Saved 128-dim projected embeddings to {proj_emb_path}")
        else:
            print("Warning: Could not extract CF embeddings, skipping projection training")
    else:
        print(f"Warning: MF checkpoint not found at {mf_ckpt}, skipping projection training")

    # Cleanup
    del model, text_embeddings
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()