"""PyTorch training loop with MPS memory guards and FP16 mixed precision."""

import gc
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, IterableDataset


DataLoaderLike = Union[DataLoader, Iterable]


class MFDataset(IterableDataset):
    """IterableDataset for Matrix Factorization training data."""

    def __init__(
        self,
        data_iter: Iterator[Dict[str, torch.Tensor]],
        n_users: int,
        n_items: int,
        implicit: bool = False,
        neg_sampling_ratio: int = 4,
    ):
        self.data_iter = data_iter
        self.n_users = n_users
        self.n_items = n_items
        self.implicit = implicit
        self.neg_sampling_ratio = neg_sampling_ratio

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.implicit:
            yield from self._implicit_iter()
        else:
            yield from self._explicit_iter()

    def _explicit_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.data_iter:
            yield {
                "user_ids": batch["userId"],
                "item_ids": batch["movieId"],
                "ratings": batch["rating"].float(),
            }

    def _implicit_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.data_iter:
            user_ids = batch["userId"]
            pos_item_ids = batch["movieId"]

            neg_items = torch.randint(
                0, self.n_items, (len(user_ids) * self.neg_sampling_ratio,), device=user_ids.device
            )
            user_ids_expanded = user_ids.repeat_interleave(self.neg_sampling_ratio)
            pos_items_expanded = pos_item_ids.repeat_interleave(self.neg_sampling_ratio)

            yield {
                "user_ids": user_ids_expanded,
                "pos_item_ids": pos_items_expanded,
                "neg_item_ids": neg_items,
            }


class MFTrainer:
    """Trainer for Matrix Factorization with MPS memory management."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoaderLike,
        val_loader: Optional[DataLoaderLike] = None,
        config: Optional[Dict[str, Any]] = None,
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

        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            foreach=False,
        )
        self.scaler = GradScaler("mps", enabled=self.use_fp16)

        self.step_count = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints/matrix_factorization"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with FP16 and gradient accumulation."""
        user_ids = batch["user_ids"].to(self.device, non_blocking=True)

        if "ratings" in batch:
            item_ids = batch["item_ids"].to(self.device, non_blocking=True)
            ratings = batch["ratings"].to(self.device, non_blocking=True)
            with autocast("mps", dtype=torch.float16, enabled=self.use_fp16):
                preds = self.model(user_ids, item_ids)
                loss = F.mse_loss(preds, ratings)
        else:
            item_ids = None  # implicit batches carry pos/neg item ids instead
            pos_item_ids = batch["pos_item_ids"].to(self.device, non_blocking=True)
            neg_item_ids = batch["neg_item_ids"].to(self.device, non_blocking=True)
            with autocast("mps", dtype=torch.float16, enabled=self.use_fp16):
                loss = self.model(user_ids, pos_item_ids, neg_item_ids)

        loss = loss / self.accum_steps
        self.scaler.scale(loss).backward()

        self.step_count += 1

        if self.step_count % self.accum_steps == 0:
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # set_to_none=False keeps the ~70MB embedding grad buffers allocated;
            # re-requesting them from the Metal driver every step OOMs on 8GB when
            # other apps hold unified memory near the working-set cap. Periodic
            # cleanup runs every 100 batches in train_epoch instead of per step.
            self.optimizer.zero_grad(set_to_none=False)

        detached_loss = loss.detach() * self.accum_steps
        del batch, user_ids, item_ids, loss

        return detached_loss

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            loss = self.train_step(batch)
            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f"  [Batch {num_batches}] Running loss: {total_loss/num_batches:.4f}")
                torch.mps.empty_cache()
                gc.collect()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch train loss: {avg_loss:.4f} | Batches: {num_batches}")
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            print("  No validation loader, skipping validation")
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            user_ids = batch["user_ids"].to(self.device, non_blocking=True)
            item_ids = batch["item_ids"].to(self.device, non_blocking=True)
            ratings = batch["ratings"].to(self.device, non_blocking=True)

            with autocast("mps", dtype=torch.float16, enabled=self.use_fp16):
                # ImplicitMF.forward computes BPR loss over triples; its pairwise
                # scorer is predict(). Explicit MF scores directly via forward.
                if hasattr(self.model, "predict"):
                    preds = self.model.predict(user_ids, item_ids)
                else:
                    preds = self.model(user_ids, item_ids)
                loss = F.mse_loss(preds, ratings)

            total_loss += loss.item()
            num_batches += 1

            del batch, user_ids, item_ids, ratings, preds, loss
            torch.mps.empty_cache()
            gc.collect()

        val_loss = total_loss / max(num_batches, 1)
        print(f"  Validation loss: {val_loss:.4f} | RMSE: {val_loss**0.5:.4f} | Batches: {num_batches}")
        return {"val_loss": val_loss, "val_rmse": val_loss**0.5}

    def train(self) -> Dict[str, list]:
        """Full training loop."""
        history = {"train_loss": [], "val_loss": [], "val_rmse": []}

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics.get("val_loss", 0))
            history["val_rmse"].append(val_metrics.get("val_rmse", 0))

            print(
                f"Epoch {epoch + 1}/{self.max_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics.get('val_loss', 0):.4f} | "
                f"Val RMSE: {val_metrics.get('val_rmse', 0):.4f}"
            )

            if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best.pt")

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

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]


def create_dataloader(
    dataset: IterableDataset,
    batch_size: int = 2048,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader with MPS-optimized settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )


def load_config(path: str = "configs/model.yaml") -> Dict[str, Any]:
    """Load model configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def train_mf(
    train_dataset: IterableDataset,
    val_dataset: Optional[IterableDataset] = None,
    config_path: str = "configs/model.yaml",
    n_users: int = 162541,
    n_items: int = 59047,
    implicit: bool = False,
) -> MFTrainer:
    """Convenience function to train MF model."""
    from src.models.matrix_factorization import MatrixFactorization

    config = load_config(config_path)

    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config.get("embedding_dim", 128),
        use_bias=not implicit,
        sparse=config.get("sparse_embeddings", False),
    )

    train_loader = create_dataloader(train_dataset, config.get("batch_size", 2048))
    val_loader = create_dataloader(val_dataset, config.get("batch_size", 2048)) if val_dataset else None

    trainer = MFTrainer(model, train_loader, val_loader, config)
    trainer.train()

    return trainer