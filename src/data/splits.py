"""Generate train/val/test splits from MovieLens Parquet ratings."""

import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


def load_config(config_path: str = "configs/data.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class SplitGenerator:
    """Generate train/val/test splits from ratings Parquet files."""

    def __init__(self, config: Dict):
        self.config = config
        self.parquet_dir = Path(config["parquet"]["output_dir"]) / "ratings"
        self.output_dir = Path(config["splits"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_cutoff = self._parse_date(config["splits"]["time_split"]["train_cutoff"])
        self.val_cutoff = self._parse_date(config["splits"]["time_split"]["val_cutoff"])
        self.test_cutoff = self._parse_date(config["splits"]["time_split"]["test_cutoff"])

        self.holdout_per_user = config["splits"]["loo_split"]["holdout_per_user"]
        self.min_interactions = config["splits"]["loo_split"]["min_interactions_per_user"]

    @staticmethod
    def _parse_date(date_str: str) -> int:
        import datetime
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def generate_time_splits(self) -> Dict[str, Path]:
        """Generate time-based train/val/test splits."""
        print("Generating time-based splits...")

        dataset = ds.dataset(self.parquet_dir, format="parquet")

        # Train: before train_cutoff
        train_filter = ds.field("timestamp") < self.train_cutoff
        # Val: train_cutoff <= timestamp < val_cutoff
        val_filter = (ds.field("timestamp") >= self.train_cutoff) & (ds.field("timestamp") < self.val_cutoff)
        # Test: val_cutoff <= timestamp < test_cutoff
        test_filter = (ds.field("timestamp") >= self.val_cutoff) & (ds.field("timestamp") < self.test_cutoff)

        splits = {
            "train": (self.output_dir / "train.parquet", train_filter),
            "val_time": (self.output_dir / "val_time.parquet", val_filter),
            "test_time": (self.output_dir / "test_time.parquet", test_filter),
        }

        for name, (out_path, filter_expr) in splits.items():
            print(f"  Writing {name}...")
            scanner = dataset.scanner(filter=filter_expr, columns=["userId", "movieId", "rating", "timestamp"])
            table = scanner.to_table()
            pq.write_table(table, out_path, compression="zstd", compression_level=3)
            print(f"    {name}: {table.num_rows:,} rows")
            del table
            gc.collect()

        return {name: path for name, (path, _) in splits.items()}

    def generate_loo_splits(self) -> Dict[str, Path]:
        """Generate leave-one-out splits per user."""
        print("Generating leave-one-out splits...")

        dataset = ds.dataset(self.parquet_dir, format="parquet")

        # First pass: collect all user interactions grouped by user
        scanner = dataset.scanner(columns=["userId", "movieId", "rating", "timestamp"])
        table = scanner.to_table()

        user_ids = table.column("userId").to_numpy()
        movie_ids = table.column("movieId").to_numpy()
        ratings = table.column("rating").to_numpy()
        timestamps = table.column("timestamp").to_numpy()

        # Group by user
        from collections import defaultdict
        user_interactions = defaultdict(list)
        for i in range(len(user_ids)):
            user_interactions[user_ids[i]].append((movie_ids[i], ratings[i], timestamps[i], i))

        del table, user_ids, movie_ids, ratings, timestamps
        gc.collect()

        # Filter users with minimum interactions
        eligible_users = {
            uid: interactions
            for uid, interactions in user_interactions.items()
            if len(interactions) >= self.min_interactions
        }

        print(f"  Eligible users: {len(eligible_users):,} (min {self.min_interactions} interactions)")

        # Sort each user's interactions by timestamp
        for uid in eligible_users:
            eligible_users[uid].sort(key=lambda x: x[2])  # sort by timestamp

        # Create train/val_loo/test_loo splits
        train_rows = []
        val_loo_rows = []
        test_loo_rows = []

        for uid, interactions in eligible_users.items():
            n = len(interactions)
            if n < self.min_interactions:
                continue

            # Test: last interaction
            test_interactions = interactions[-self.holdout_per_user:]
            # Val: second-to-last interaction
            val_interactions = interactions[-(self.holdout_per_user + 1):-self.holdout_per_user]
            # Train: everything else
            train_interactions = interactions[:-(self.holdout_per_user + 1)]

            for mid, rating, ts, _ in train_interactions:
                train_rows.append((uid, mid, rating, ts))
            for mid, rating, ts, _ in val_interactions:
                val_loo_rows.append((uid, mid, rating, ts))
            for mid, rating, ts, _ in test_interactions:
                test_loo_rows.append((uid, mid, rating, ts))

        print(f"  Train rows: {len(train_rows):,}")
        print(f"  Val LOO rows: {len(val_loo_rows):,}")
        print(f"  Test LOO rows: {len(test_loo_rows):,}")

        # Write splits
        schema = pa.schema([
            pa.field("userId", pa.int32()),
            pa.field("movieId", pa.int32()),
            pa.field("rating", pa.float32()),
            pa.field("timestamp", pa.int64()),
        ])

        splits = {
            "train_loo": (self.output_dir / "train_loo.parquet", train_rows),
            "val_loo": (self.output_dir / "val_loo.parquet", val_loo_rows),
            "test_loo": (self.output_dir / "test_loo.parquet", test_loo_rows),
        }

        for name, (out_path, rows) in splits.items():
            print(f"  Writing {name}...")
            if rows:
                table = pa.Table.from_arrays(
                    [pa.array([r[i] for r in rows], type=schema.field(i).type) for i in range(4)],
                    schema=schema,
                )
                pq.write_table(table, out_path, compression="zstd", compression_level=3)
            else:
                # Write empty table with schema
                table = pa.Table.from_arrays([pa.array([], type=schema.field(i).type) for i in range(4)], schema=schema)
                pq.write_table(table, out_path, compression="zstd", compression_level=3)
            print(f"    {name}: {len(rows):,} rows")

        return {name: path for name, (path, _) in splits.items()}


def main():
    config = load_config()
    generator = SplitGenerator(config)

    time_splits = generator.generate_time_splits()
    loo_splits = generator.generate_loo_splits()

    print("\n✓ All splits generated:")
    for name, path in {**time_splits, **loo_splits}.items():
        pf = pq.ParquetFile(path)
        print(f"  {name}: {pf.metadata.num_rows:,} rows ({path})")


if __name__ == "__main__":
    main()