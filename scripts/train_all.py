#!/usr/bin/env python
"""Stage 2: Train Matrix Factorization or Two-Tower models.

Usage:
    python -m scripts.train_all --model mf
    python -m scripts.train_all --model mf_implicit
    python -m scripts.train_all --model two_tower
"""

import argparse
import gc
import yaml
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Iterable

from src.data.cold_start import build_aligned_metadata
from src.data.dataset import create_dataloaders
from src.models.trainer import MFTrainer, load_config, MFDataset
from src.models.matrix_factorization import MatrixFactorization, ImplicitMF
from src.models.two_tower import TwoTower, TwoTowerWithMetadata
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


def save_id_mappings(user_map: Dict[int, int], movie_map: Dict[int, int], output_dir: Path):
    """Save user and movie ID mappings to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # User mapping
    user_table = pa.Table.from_pydict({
        "userId": list(user_map.keys()),
        "user_idx": list(user_map.values()),
    })
    pq.write_table(user_table, output_dir / "user_mapping.parquet", compression="zstd")
    
    # Movie mapping
    movie_table = pa.Table.from_pydict({
        "movieId": list(movie_map.keys()),
        "movie_idx": list(movie_map.values()),
    })
    pq.write_table(movie_table, output_dir / "movie_mapping.parquet", compression="zstd")
    
    print(f"Saved mappings: {len(user_map)} users, {len(movie_map)} movies -> {output_dir}")


def load_data_config(path: str = "configs/data.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_mf_explicit(args, data_config, model_config):
    """Train explicit Matrix Factorization (MSE loss)."""
    print("=" * 60)
    print("Training Matrix Factorization (Explicit / MSE)")
    print("=" * 60)

    train_loader, val_time_loader, val_loo_loader, test_loader, user_map, movie_map = create_dataloaders(
        data_config,
        batch_size=model_config.get("batch_size", 2048),
        shuffle_train=True,
    )

    n_users = len(user_map)
    n_items = len(movie_map)
    print(f"Users: {n_users:,}, Items: {n_items:,}")

    save_id_mappings(user_map, movie_map, Path(data_config.get("splits", {}).get("output_dir", "data/processed")))

    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=model_config.get("embedding_dim", 128),
        use_bias=model_config.get("use_bias", True),
        sparse=model_config.get("sparse_embeddings", False),
    )

    checkpoint_dir = model_config.get("checkpoint_dir", "checkpoints/matrix_factorization")
    model_config["checkpoint_dir"] = checkpoint_dir

    train_loader = MFDataset(train_loader, n_users, n_items, implicit=False)
    val_time_loader = MFDataset(val_time_loader, n_users, n_items, implicit=False)

    trainer = MFTrainer(model, train_loader, val_time_loader, model_config)
    history = trainer.train()

    print(f"\nTraining complete. Best val loss: {trainer.best_val_loss:.4f}")
    return history


def train_mf_implicit(args, data_config, model_config):
    """Train implicit Matrix Factorization (BPR loss)."""
    print("=" * 60)
    print("Training Matrix Factorization (Implicit / BPR)")
    print("=" * 60)

    train_loader, val_time_loader, val_loo_loader, test_loader, user_map, movie_map = create_dataloaders(
        data_config,
        batch_size=model_config.get("batch_size", 2048),
        shuffle_train=True,
    )

    n_users = len(user_map)
    n_items = len(movie_map)
    print(f"Users: {n_users:,}, Items: {n_items:,}")

    save_id_mappings(user_map, movie_map, Path(data_config.get("splits", {}).get("output_dir", "data/processed")))

    model = ImplicitMF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=model_config.get("embedding_dim", 128),
        sparse=model_config.get("sparse_embeddings", False),
    )

    checkpoint_dir = model_config.get("checkpoint_dir", "checkpoints/matrix_factorization_implicit")
    model_config["checkpoint_dir"] = checkpoint_dir

    train_loader = MFDataset(train_loader, n_users, n_items, implicit=True)
    val_time_loader = MFDataset(val_time_loader, n_users, n_items, implicit=False)

    trainer = MFTrainer(model, train_loader, val_time_loader, model_config)
    history = trainer.train()

    print(f"\nTraining complete. Best val loss: {trainer.best_val_loss:.4f}")
    return history


class TwoTowerDataset(torch.utils.data.IterableDataset):
    """IterableDataset for Two-Tower training with metadata and negative sampling."""

    def __init__(
        self,
        data_iter: Iterator[Dict[str, torch.Tensor]],
        n_users: int,
        n_items: int,
        item_metadata: torch.Tensor,
        neg_sampling_ratio: int = 4,
        implicit: bool = True,
    ):
        self.data_iter = data_iter
        self.n_users = n_users
        self.n_items = n_items
        self.item_metadata = item_metadata
        self.neg_sampling_ratio = neg_sampling_ratio
        self.implicit = implicit

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.data_iter:
            user_ids = batch["userId"]
            pos_item_ids = batch["movieId"]

            if self.implicit:
                neg_item_ids = torch.randint(
                    0, self.n_items,
                    (len(user_ids) * self.neg_sampling_ratio,),
                    device=user_ids.device,
                    dtype=pos_item_ids.dtype,
                )
                user_ids_expanded = user_ids.repeat_interleave(self.neg_sampling_ratio)
                pos_items_expanded = pos_item_ids.repeat_interleave(self.neg_sampling_ratio)

                pos_metadata = self.item_metadata[pos_items_expanded]
                neg_metadata = self.item_metadata[neg_item_ids]

                yield {
                    "user_ids": user_ids_expanded,
                    "pos_item_ids": pos_items_expanded,
                    "neg_item_ids": neg_item_ids,
                    "pos_item_metadata": pos_metadata,
                    "neg_item_metadata": neg_metadata,
                }
            else:
                ratings = batch["rating"].float()
                pos_metadata = self.item_metadata[pos_item_ids]

                yield {
                    "user_ids": user_ids,
                    "pos_item_ids": pos_item_ids,
                    "pos_item_metadata": pos_metadata,
                    "ratings": ratings,
                }


class TwoTowerTrainer:
    """Trainer for Two-Tower model with MPS memory management, FP16, LR scheduling, and early stopping."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Iterable,
        val_loader: Optional[Iterable],
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config or {}
        self.batch_size = self.config.get("batch_size", 2048)
        self.accum_steps = self.config.get("accum_steps", 4)
        self.lr = float(self.config.get("lr", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 1e-5))
        self.max_epochs = self.config.get("max_epochs", 10)
        self.use_fp16 = self.config.get("use_fp16", True)
        self.grad_clip = float(self.config.get("grad_clip", 1.0))
        self.loss_type = self.config.get("loss_type", "bce")

        # LR scheduler config
        lr_sched_cfg = self.config.get("lr_scheduler", {})
        self.lr_scheduler_type = lr_sched_cfg.get("type", "reduce_on_plateau")
        self.lr_scheduler_patience = lr_sched_cfg.get("patience", 3)
        self.lr_scheduler_factor = lr_sched_cfg.get("factor", 0.5)
        self.lr_scheduler_min_lr = lr_sched_cfg.get("min_lr", 1e-6)

        # Early stopping config
        es_cfg = self.config.get("early_stopping", {})
        self.early_stopping_patience = es_cfg.get("patience", 5)
        self.early_stopping_min_delta = es_cfg.get("min_delta", 1e-4)

        # Logging config
        self.log_interval = self.config.get("log_interval", 500)

        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            foreach=False,
        )
        self.scaler = GradScaler("mps", enabled=self.use_fp16)

        # Initialize LR scheduler
        if self.lr_scheduler_type == "reduce_on_plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=self.lr_scheduler_min_lr,
            )
        else:
            self.lr_scheduler = None

        self.step_count = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Early stopping state
        self.epochs_no_improve = 0
        self.es_best_loss = float("inf")

        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints/two_tower"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _bce_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss for retrieval (positive=1, negative=0)."""
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_scores, pos_labels)
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_scores, neg_labels)

        return pos_loss + neg_loss

    def _margin_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
        """Contrastive margin loss (max(0, margin - pos_score + neg_score))."""
        return torch.clamp(margin - pos_scores.unsqueeze(1) + neg_scores, min=0).mean()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with FP16 and gradient accumulation."""
        user_ids = batch["user_ids"].to(self.device, non_blocking=True)
        pos_item_ids = batch["pos_item_ids"].to(self.device, non_blocking=True)
        pos_item_metadata = batch["pos_item_metadata"].to(self.device, non_blocking=True)
        neg_item_ids = batch["neg_item_ids"].to(self.device, non_blocking=True)
        neg_item_metadata = batch["neg_item_metadata"].to(self.device, non_blocking=True)

        with autocast("mps", dtype=torch.float16, enabled=self.use_fp16):
            pos_scores = self.model(user_ids, pos_item_ids, pos_item_metadata)
            neg_scores = self.model(user_ids, neg_item_ids, neg_item_metadata)

            if self.loss_type == "bce":
                loss = self._bce_loss(pos_scores, neg_scores)
            else:
                loss = self._margin_loss(pos_scores, neg_scores)

        loss = loss / self.accum_steps
        self.scaler.scale(loss).backward()

        self.step_count += 1

        if self.step_count % self.accum_steps == 0:
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        detached_loss = loss.detach() * self.accum_steps
        del batch, user_ids, pos_item_ids, pos_item_metadata, neg_item_ids, neg_item_metadata, loss
        # Cleanup runs every 100 batches in train_epoch, not per step: empty_cache() per step
        # forces the allocator to return blocks it immediately reallocates, and the del above
        # already drops the refcounts (gc.collect only matters for reference cycles).
        return detached_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation using BCE loss on held-out interactions."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            user_ids = batch["user_ids"].to(self.device, non_blocking=True)
            pos_item_ids = batch["pos_item_ids"].to(self.device, non_blocking=True)
            pos_item_metadata = batch["pos_item_metadata"].to(self.device, non_blocking=True)
            neg_item_ids = batch["neg_item_ids"].to(self.device, non_blocking=True)
            neg_item_metadata = batch["neg_item_metadata"].to(self.device, non_blocking=True)

            with autocast("mps", dtype=torch.float16, enabled=self.use_fp16):
                pos_scores = self.model(user_ids, pos_item_ids, pos_item_metadata)
                neg_scores = self.model(user_ids, neg_item_ids, neg_item_metadata)

                if self.loss_type == "bce":
                    loss = self._bce_loss(pos_scores, neg_scores)
                else:
                    loss = self._margin_loss(pos_scores, neg_scores)

            total_loss += loss.item()
            num_batches += 1

            del batch, user_ids, pos_item_ids, pos_item_metadata, neg_item_ids, neg_item_metadata, loss
            if num_batches % 100 == 0:
                torch.mps.empty_cache()
                gc.collect()

        val_loss = total_loss / max(num_batches, 1)
        return {"val_loss": val_loss}

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with detailed logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        running_loss = 0.0

        for batch in self.train_loader:
            loss = self.train_step(batch)
            total_loss += loss.item()
            running_loss += loss.item()
            num_batches += 1

            if num_batches % self.log_interval == 0:
                avg_running = running_loss / self.log_interval
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {self.epoch + 1}/{self.max_epochs} | "
                    f"Batch {num_batches} | "
                    f"Running Loss: {avg_running:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                running_loss = 0.0

            if num_batches % 100 == 0:
                torch.mps.empty_cache()
                gc.collect()

        return {"train_loss": total_loss / max(num_batches, 1)}

    def train(self) -> Dict[str, list]:
        """Full training loop with LR scheduling and early stopping."""
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics.get("val_loss", 0))

            print(
                f"Epoch {epoch + 1}/{self.max_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics.get('val_loss', 0):.4f}"
            )

            # LR scheduler step
            val_loss = val_metrics.get("val_loss", float("inf"))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.es_best_loss - self.early_stopping_min_delta:
                self.es_best_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                print(f"  Early stopping: {self.epochs_no_improve}/{self.early_stopping_patience} epochs without improvement")
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print(f"  Early stopping triggered after {epoch + 1} epochs")
                    break

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")
                self.save_inference_checkpoint("best_model.pt")

            self.save_checkpoint(f"epoch_{epoch}.pt")

            torch.mps.empty_cache()
            gc.collect()

        return history

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "step_count": self.step_count,
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )

    def save_inference_checkpoint(self, name: str) -> None:
        """Save weights only — this is the artifact evaluate.py, build_index.py, and the
        API load. Kept separate from best.pt so serving never pulls in optimizer state.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            self.checkpoint_dir / name,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]


def train_two_tower(args, data_config, model_config):
    """Train Two-Tower model with metadata support using cold-start embeddings."""
    print("=" * 60)
    print("Training Two-Tower Retrieval Model")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, val_time_loader, val_loo_loader, test_loader, user_map, movie_map = create_dataloaders(
        data_config,
        batch_size=model_config.get("batch_size", 2048),
        shuffle_train=True,
        device=device,
    )

    n_users = len(user_map)
    n_items = len(movie_map)
    print(f"Users: {n_users:,}, Items: {n_items:,}")

    processed_dir = Path(data_config.get("splits", {}).get("output_dir", "data/processed"))
    save_id_mappings(user_map, movie_map, processed_dir)

    item_metadata_np = build_aligned_metadata(movie_map, processed_dir)

    item_metadata = torch.from_numpy(item_metadata_np).to(device)
    metadata_dim = item_metadata.shape[1]  # 128

    # Architecture params live under the `two_tower:` block in configs/model.yaml
    tt_config = model_config.get("two_tower", {})

    # Initialize TwoTowerWithMetadata model
    model = TwoTowerWithMetadata(
        n_users=n_users,
        n_items=n_items,
        metadata_dim=metadata_dim,
        embedding_dim=model_config.get("embedding_dim", 128),
        hidden_dim=tt_config.get("hidden_dim", 256),
        output_dim=tt_config.get("output_dim", 128),
        dropout=tt_config.get("dropout", 0.1),
    )

    checkpoint_dir = model_config.get("checkpoint_dir", "checkpoints/two_tower")
    model_config["checkpoint_dir"] = checkpoint_dir

    # Wrap loaders with negative sampling and metadata lookup
    neg_ratio = tt_config.get("neg_sampling_ratio", 4)

    train_dataset = TwoTowerDataset(
        train_loader,
        n_users=n_users,
        n_items=n_items,
        item_metadata=item_metadata,
        neg_sampling_ratio=neg_ratio,
        implicit=True,
    )
    val_dataset = TwoTowerDataset(
        val_time_loader,
        n_users=n_users,
        n_items=n_items,
        item_metadata=item_metadata,
        neg_sampling_ratio=neg_ratio,
        implicit=True,
    )

    # Training configuration
    train_config = {
        "batch_size": model_config.get("batch_size", 2048),
        "accum_steps": model_config.get("accum_steps", 4),
        "lr": model_config.get("lr", 0.001),
        "weight_decay": model_config.get("weight_decay", 1e-5),
        "max_epochs": model_config.get("max_epochs", 10),
        "use_fp16": model_config.get("use_fp16", True),
        "grad_clip": model_config.get("grad_clip", 1.0),
        "loss_type": model_config.get("loss_type", "bce"),
        "checkpoint_dir": checkpoint_dir,
        "neg_sampling_ratio": neg_ratio,
        "lr_scheduler": model_config.get("lr_scheduler", {}),
        "early_stopping": model_config.get("early_stopping", {}),
        "log_interval": model_config.get("log_interval", 500),
        # Model architecture params saved in checkpoint for evaluate.py / build_index.py
        "n_users": n_users,
        "n_items": n_items,
        "metadata_dim": metadata_dim,
        "embedding_dim": model_config.get("embedding_dim", 128),
        "hidden_dim": tt_config.get("hidden_dim", 256),
        "output_dim": tt_config.get("output_dim", 128),
        "dropout": tt_config.get("dropout", 0.1),
    }

    trainer = TwoTowerTrainer(model, train_dataset, val_dataset, train_config, device)
    history = trainer.train()

    print(f"\nTraining complete. Best val loss: {trainer.best_val_loss:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mf", "mf_implicit", "two_tower"],
        required=True,
        help="Model to train: mf (explicit), mf_implicit (BPR), or two_tower"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config"
    )
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    model_config = load_config(args.model_config)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    if args.model == "mf":
        train_mf_explicit(args, data_config, model_config)
    elif args.model == "mf_implicit":
        train_mf_implicit(args, data_config, model_config)
    elif args.model == "two_tower":
        train_two_tower(args, data_config, model_config)

    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()