"""Streaming IterableDataset for MovieLens Parquet data.

Memory-efficient data loading using PyArrow dataset scanning with optional filtering.
"""

import gc
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


class ParquetStreamingDataset(IterableDataset):
    """IterableDataset that streams Parquet files in chunks without loading full data to RAM."""

    def __init__(
        self,
        parquet_dir: Union[str, Path],
        columns: Optional[List[str]] = None,
        filter_expr: Optional[ds.Expression] = None,
        batch_size: int = 1024,
        shuffle: bool = False,
        shuffle_buffer_size: int = 1_000_000,
        seed: int = 42,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.columns = columns
        self.filter_expr = filter_expr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size  # rows, not batches
        self.seed = seed
        self._pass_count = 0

        # Discover all parquet files (handles both single file and directory)
        path = Path(parquet_dir)
        if path.is_file():
            self.files = [path]
        elif path.is_dir():
            self.files = sorted(path.glob("*.parquet"))
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not self.files:
            raise FileNotFoundError(f"No parquet files found in {path}")

        # Build PyArrow dataset for efficient scanning
        self.dataset = ds.dataset(self.files, format="parquet")

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield batches as dict of numpy arrays."""
        scanner = self.dataset.scanner(
            columns=self.columns,
            filter=self.filter_expr,
            batch_size=self.batch_size,
        )

        if self.shuffle:
            yield from self._shuffled_iter(scanner)
        else:
            yield from self._sequential_iter(scanner)

    def _sequential_iter(self, scanner) -> Iterator[Dict[str, np.ndarray]]:
        for batch in scanner.to_batches():
            yield self._batch_to_dict(batch)

    def _shuffled_iter(self, scanner) -> Iterator[Dict[str, np.ndarray]]:
        """Row-level windowed shuffle: fill a buffer of rows, permute, emit as batches.

        Parquet rows are grouped by user, so shuffling whole scanner batches (the old
        behavior) kept each emitted batch single-user — fatal for in-batch negatives.
        The seed advances every pass so epochs see different orders.
        """
        rng = np.random.default_rng((self.seed, self._pass_count))
        self._pass_count += 1

        buffer: List[Dict[str, np.ndarray]] = []
        buffered_rows = 0

        def drain() -> Iterator[Dict[str, np.ndarray]]:
            if not buffer:
                return
            merged = {k: np.concatenate([b[k] for b in buffer]) for k in buffer[0]}
            perm = rng.permutation(len(next(iter(merged.values()))))
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                yield {k: v[idx] for k, v in merged.items()}

        for batch in scanner.to_batches():
            batch_dict = self._batch_to_dict(batch)
            buffer.append(batch_dict)
            buffered_rows += len(next(iter(batch_dict.values())))

            if buffered_rows >= self.shuffle_buffer_size:
                yield from drain()
                buffer, buffered_rows = [], 0

        yield from drain()

    def _batch_to_dict(self, batch: pa.RecordBatch) -> Dict[str, np.ndarray]:
        """Convert PyArrow RecordBatch to dict of numpy arrays (zero-copy where possible)."""
        result = {}
        for col in batch.schema.names:
            arr = batch.column(col)
            # Zero-copy to numpy for primitive types
            if pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type):
                result[col] = arr.to_numpy(zero_copy_only=False)
            else:
                result[col] = arr.to_pylist()
        return result


class RatingsDataset(ParquetStreamingDataset):
    """Streaming dataset for ratings with user/movie ID remapping support."""

    def __init__(
        self,
        parquet_dir: Union[str, Path],
        user_id_map: Optional[Dict[int, int]] = None,
        movie_id_map: Optional[Dict[int, int]] = None,
        batch_size: int = 1024,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(
            parquet_dir,
            columns=["userId", "movieId", "rating", "timestamp"],
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        for batch in super().__iter__():
            if self.user_id_map is not None:
                batch["userId"] = np.vectorize(self.user_id_map.get)(batch["userId"])
            if self.movie_id_map is not None:
                batch["movieId"] = np.vectorize(self.movie_id_map.get)(batch["movieId"])
            yield batch


def build_id_mappings(parquet_dir: Union[str, Path]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build contiguous 0-based ID mappings from Parquet files."""
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))

    # Only read ratings files (train/val/test), exclude cold_start_corpus
    rating_files = [f for f in files if f.name.startswith(("train", "val", "test"))]

    user_ids = set()
    movie_ids = set()

    for f in rating_files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["userId", "movieId"]):
            user_ids.update(batch.column("userId").to_pylist())
            movie_ids.update(batch.column("movieId").to_pylist())

    user_id_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    movie_id_map = {mid: idx for idx, mid in enumerate(sorted(movie_ids))}

    return user_id_map, movie_id_map


class MemoryEfficientDataLoader:
    """Wrapper around IterableDataset with explicit memory management."""

    def __init__(
        self,
        dataset: IterableDataset,
        device: torch.device,
        prefetch_batches: int = 2,
    ):
        self.dataset = dataset
        self.device = device
        self.prefetch_batches = prefetch_batches
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self.dataset)
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._iterator is None:
            raise StopIteration
        batch = next(self._iterator)
        # Convert to tensors on device
        tensor_batch = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                tensor_batch[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
            else:
                tensor_batch[k] = torch.tensor(v, device=self.device)
        return tensor_batch

    def __del__(self):
        if self._iterator is not None:
            del self._iterator
            gc.collect()


def create_dataloaders(
    config: Dict,
    batch_size: int = 1024,
    shuffle_train: bool = True,
    device: Optional[torch.device] = None,
    protocol: str = "time",
) -> Tuple[
    MemoryEfficientDataLoader,
    MemoryEfficientDataLoader,
    Dict[int, int],
    Dict[int, int],
]:
    """Create train/val dataloaders from processed Parquet splits.

    protocol="time" trains on the time split (train/val_time); protocol="loo" trains
    on the leave-one-out split (train_loo/val_loo). Training and evaluation must use
    the same protocol: evaluate.py tests on test_loo, so a model trained on the time
    split has the LOO test answers inside its training data.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    splits_dir = Path(config["splits"]["output_dir"])

    train_file, val_file = {
        "time": ("train.parquet", "val_time.parquet"),
        "loo": ("train_loo.parquet", "val_loo.parquet"),
    }[protocol]

    # ID mappings cover all splits (build_id_mappings reads every train/val/test file),
    # so both protocols share the same user/item index space.
    user_map, movie_map = build_id_mappings(splits_dir)

    train_ds = RatingsDataset(
        splits_dir / train_file,
        user_id_map=user_map,
        movie_id_map=movie_map,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_ds = RatingsDataset(
        splits_dir / val_file,
        user_id_map=user_map,
        movie_id_map=movie_map,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        MemoryEfficientDataLoader(train_ds, device),
        MemoryEfficientDataLoader(val_ds, device),
        user_map,
        movie_map,
    )