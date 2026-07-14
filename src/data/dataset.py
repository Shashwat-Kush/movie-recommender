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
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.columns = columns
        self.filter_expr = filter_expr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

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
        """Shuffle using reservoir sampling buffer."""
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)

        buffer = []
        for batch in scanner.to_batches():
            batch_dict = self._batch_to_dict(batch)
            buffer.append(batch_dict)

            if len(buffer) >= self.shuffle_buffer_size:
                idx = random.randrange(len(buffer))
                yield buffer.pop(idx)

        # Flush remaining buffer
        random.shuffle(buffer)
        for batch_dict in buffer:
            yield batch_dict

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
) -> Tuple[
    MemoryEfficientDataLoader,
    MemoryEfficientDataLoader,
    MemoryEfficientDataLoader,
    MemoryEfficientDataLoader,
    Dict[int, int],
    Dict[int, int],
]:
    """Create train/val/test dataloaders from processed Parquet splits."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    splits_dir = Path(config["splits"]["output_dir"])

    # Build ID mappings from training data
    train_dir = splits_dir / "train.parquet"
    if train_dir.is_dir():
        user_map, movie_map = build_id_mappings(train_dir)
    else:
        # Single file
        user_map, movie_map = build_id_mappings(train_dir.parent)

    train_ds = RatingsDataset(
        splits_dir / "train.parquet",
        user_id_map=user_map,
        movie_id_map=movie_map,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_time_ds = RatingsDataset(
        splits_dir / "val_time.parquet",
        user_id_map=user_map,
        movie_id_map=movie_map,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loo_ds = RatingsDataset(
        splits_dir / "val_loo.parquet",
        user_id_map=user_map,
        movie_id_map=movie_map,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = RatingsDataset(
        splits_dir / "test_time.parquet",
        user_id_map=user_map,
        movie_id_map=movie_map,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        MemoryEfficientDataLoader(train_ds, device),
        MemoryEfficientDataLoader(val_time_ds, device),
        MemoryEfficientDataLoader(val_loo_ds, device),
        MemoryEfficientDataLoader(test_ds, device),
        user_map,
        movie_map,
    )