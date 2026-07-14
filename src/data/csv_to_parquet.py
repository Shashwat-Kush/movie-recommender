"""Chunked CSV to Parquet converter for MovieLens 25M dataset."""

import gc
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


class MemoryMonitor:
    """Track memory usage and trigger GC when threshold exceeded."""

    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        self._process = None
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
        except ImportError:
            pass

    def check_and_collect(self) -> float:
        if self._process is None:
            return 0.0
        mem_mb = self._process.memory_info().rss / (1024 * 1024)
        if mem_mb > self.max_memory_mb * 0.8:
            gc.collect()
            mem_mb = self._process.memory_info().rss / (1024 * 1024)
        return mem_mb


def load_config(config_path: str = "configs/data.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_arrow_schema(schema_config: Dict[str, str]) -> pa.Schema:
    type_map = {
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "string": pa.string(),
        "bool": pa.bool_(),
    }
    fields = [pa.field(name, type_map[dtype]) for name, dtype in schema_config.items()]
    return pa.schema(fields)


SCHEMAS = {
    "ratings": {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64",
    },
    "movies": {
        "movieId": "int32",
        "title": "string",
        "genres": "string",
    },
    "tags": {
        "userId": "int32",
        "movieId": "int32",
        "tag": "string",
        "timestamp": "int64",
    },
    "genome_scores": {
        "movieId": "int32",
        "tagId": "int32",
        "relevance": "float32",
    },
    "genome_tags": {
        "tagId": "int32",
        "tag": "string",
    },
    "links": {
        "movieId": "int32",
        "imdbId": "float32",
        "tmdbId": "float32",
    },
}

PANDAS_DTYPES = {
    "ratings": {"userId": "int32", "movieId": "int32", "rating": "float32"},
    "movies": {"movieId": "int32", "title": "string", "genres": "string"},
    "tags": {"userId": "int32", "movieId": "int32", "tag": "string"},
    "genome_scores": {"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    "genome_tags": {"tagId": "int32", "tag": "string"},
    "links": {"movieId": "int32", "imdbId": "float32", "tmdbId": "float32"},
}


PARSE_DATES = {
    "ratings": ["timestamp"],
    "tags": ["timestamp"],
    "movies": [],
    "genome_scores": [],
    "genome_tags": [],
    "links": [],
}


def csv_chunk_iterator(
    csv_path: Path,
    chunk_size: int,
    dtypes: Optional[Dict] = None,
    parse_dates: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype=dtypes,
        parse_dates=parse_dates,
        low_memory=False,
        memory_map=True,
    ):
        # Convert datetime columns to Unix timestamp (int64 seconds)
        if parse_dates:
            for col in parse_dates:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype("int64") // 10**9
        
        # Handle nullable integer columns (NaN -> 0)
        for col in chunk.columns:
            if chunk[col].dtype == "float64" and col in ["tmdbId"]:
                chunk[col] = chunk[col].fillna(0).astype("int32")
        
        yield chunk


def write_chunk_to_parquet(
    chunk: pd.DataFrame,
    output_dir: Path,
    partition_name: str,
    chunk_idx: int,
    schema: pa.Schema,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 50000,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_file = output_dir / f"{partition_name}-part-{chunk_idx:04d}.parquet"

    table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
    pq.write_table(
        table,
        partition_file,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        use_dictionary=True,
        write_statistics=True,
    )
    return partition_file


def convert_csv_to_parquet(
    config: Dict,
    file_key: str,
    monitor: Optional[MemoryMonitor] = None,
) -> List[Path]:
    raw_config = config["raw"]
    parquet_config = config["parquet"]

    csv_path = Path(raw_config[file_key])
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    file_config = parquet_config[file_key]
    output_dir = Path(parquet_config["output_dir"]) / file_key
    chunk_size = file_config.get("chunk_size", 100000)
    compression = file_config.get("compression", "zstd")
    compression_level = file_config.get("compression_level", 3)
    row_group_size = file_config.get("row_group_size", 50000)

    schema = get_arrow_schema(SCHEMAS[file_key])
    pandas_dtypes = PANDAS_DTYPES[file_key]
    parse_dates = PARSE_DATES[file_key]

    written_files = []
    chunk_idx = 0

    print(f"Converting {csv_path.name} -> {output_dir} (chunk_size={chunk_size:,})")

    for chunk in csv_chunk_iterator(csv_path, chunk_size, dtypes=pandas_dtypes, parse_dates=parse_dates):
        if monitor:
            mem_mb = monitor.check_and_collect()
            if chunk_idx % 10 == 0:
                print(f"  Chunk {chunk_idx}: {len(chunk):,} rows, mem={mem_mb:.0f}MB")

        write_chunk_to_parquet(
            chunk,
            output_dir,
            file_key,
            chunk_idx,
            schema,
            compression,
            compression_level,
            row_group_size,
        )
        written_files.append(output_dir / f"{file_key}-part-{chunk_idx:04d}.parquet")
        chunk_idx += 1

        del chunk
        if chunk_idx % 5 == 0:
            gc.collect()

    print(f"  Completed: {chunk_idx} partitions written to {output_dir}")
    return written_files


def verify_parquet_output(config: Dict, results: Dict[str, List[Path]]) -> None:
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    for file_key, files in results.items():
        total_rows = 0
        for f in files:
            pf = pq.ParquetFile(f)
            total_rows += pf.metadata.num_rows
            print(f"  {f.name}: {pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups")
        print(f"  {file_key} TOTAL: {total_rows:,} rows")


def main():
    config = load_config()
    monitor = MemoryMonitor(max_memory_mb=config["memory"]["gc_threshold_mb"])
    results = {}

    for file_key in config["raw"]:
        print(f"\n{'='*60}")
        print(f"Processing: {file_key}")
        print(f"{'='*60}")
        results[file_key] = convert_csv_to_parquet(config, file_key, monitor)

    verify_parquet_output(config, results)
    print("\n✓ All CSV files converted to Parquet successfully")


if __name__ == "__main__":
    main()