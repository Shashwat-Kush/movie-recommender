"""Cold-start pipeline: build text corpus from movie metadata for embedding generation."""

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


def load_config(config_path: str = "configs/data.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_movies(parquet_dir: Path) -> pa.Table:
    """Load all movie parquet files into a single table."""
    dataset = ds.dataset(parquet_dir, format="parquet")
    scanner = dataset.scanner(columns=["movieId", "title", "genres"])
    return scanner.to_table()


def load_tags(parquet_dir: Path) -> pa.Table:
    """Load all tag parquet files into a single table."""
    dataset = ds.dataset(parquet_dir, format="parquet")
    scanner = dataset.scanner(columns=["movieId", "tag"])
    return scanner.to_table()


def load_genome_scores(parquet_dir: Path) -> pa.Table:
    """Load all genome scores parquet files into a single table."""
    dataset = ds.dataset(parquet_dir, format="parquet")
    scanner = dataset.scanner(columns=["movieId", "tagId", "relevance"])
    return scanner.to_table()


def load_genome_tags(parquet_dir: Path) -> pa.Table:
    """Load genome tags parquet file."""
    dataset = ds.dataset(parquet_dir, format="parquet")
    scanner = dataset.scanner(columns=["tagId", "tag"])
    return scanner.to_table()


def build_top_tags(tags_table: pa.Table, top_k: int = 5) -> Dict[int, str]:
    """Build top-k tags per movie from user tags."""
    df = tags_table.to_pandas()
    top_tags = (
        df.groupby("movieId")["tag"]
        .apply(lambda x: ", ".join(x.value_counts().head(top_k).index.tolist()))
        .to_dict()
    )
    del df
    gc.collect()
    return top_tags


def build_top_genome_tags(
    genome_scores_table: pa.Table,
    genome_tags_table: pa.Table,
    top_k: int = 10,
) -> Dict[int, str]:
    """Build top-k genome tags per movie by relevance score."""
    scores_df = genome_scores_table.to_pandas()
    tags_df = genome_tags_table.to_pandas()

    tag_map = dict(zip(tags_df["tagId"], tags_df["tag"]))

    scores_df["tag_name"] = scores_df["tagId"].map(tag_map)
    scores_df = scores_df.dropna(subset=["tag_name"])

    top_genome = (
        scores_df.sort_values("relevance", ascending=False)
        .groupby("movieId")["tag_name"]
        .apply(lambda x: ", ".join(x.head(top_k).tolist()))
        .to_dict()
    )

    del scores_df, tags_df, tag_map
    gc.collect()
    return top_genome


def build_text_corpus(
    movies_table: pa.Table,
    top_tags: Dict[int, str],
    top_genome_tags: Dict[int, str],
) -> Tuple[List[int], List[str]]:
    """Build text corpus per movie using template: '{title}. Genres: {genres}. Tags: {top_tags}. Genome: {top_genome_tags}'"""
    df = movies_table.to_pandas()
    movie_ids = df["movieId"].tolist()

    corpus = []
    for _, row in df.iterrows():
        mid = row["movieId"]
        title = row["title"] if pd.notna(row["title"]) else ""
        genres = row["genres"] if pd.notna(row["genres"]) else ""
        tags = top_tags.get(mid, "")
        genome = top_genome_tags.get(mid, "")

        text = f"{title}. Genres: {genres}. Tags: {tags}. Genome: {genome}"
        corpus.append(text)

    del df
    gc.collect()
    return movie_ids, corpus


import pandas as pd


def load_cold_start_data(
    config: Optional[Dict] = None,
    top_tags_k: int = 5,
    top_genome_k: int = 10,
) -> Tuple[List[int], List[str]]:
    """Main entry point: load all metadata and build text corpus.

    Returns:
        Tuple of (movie_ids, corpus_texts) aligned by index.
    """
    if config is None:
        config = load_config()

    parquet_root = Path(config["parquet"]["output_dir"])

    print("Loading movies...")
    movies_table = load_movies(parquet_root / "movies")
    print(f"  Loaded {movies_table.num_rows:,} movies")

    print("Loading user tags...")
    tags_table = load_tags(parquet_root / "tags")
    print(f"  Loaded {tags_table.num_rows:,} tag entries")

    print("Loading genome scores...")
    genome_scores_table = load_genome_scores(parquet_root / "genome_scores")
    print(f"  Loaded {genome_scores_table.num_rows:,} genome score entries")

    print("Loading genome tags...")
    genome_tags_table = load_genome_tags(parquet_root / "genome_tags")
    print(f"  Loaded {genome_tags_table.num_rows:,} genome tags")

    print("Building top tags per movie...")
    top_tags = build_top_tags(tags_table, top_k=top_tags_k)
    del tags_table
    gc.collect()

    print("Building top genome tags per movie...")
    top_genome_tags = build_top_genome_tags(
        genome_scores_table, genome_tags_table, top_k=top_genome_k
    )
    del genome_scores_table, genome_tags_table
    gc.collect()

    print("Building text corpus...")
    movie_ids, corpus = build_text_corpus(movies_table, top_tags, top_genome_tags)
    del movies_table
    gc.collect()

    print(f"Built corpus for {len(movie_ids):,} movies")
    return movie_ids, corpus


def save_corpus(movie_ids: List[int], corpus: List[str], output_path: Path) -> None:
    """Save movie IDs and corpus to parquet for inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict({"movieId": movie_ids, "text": corpus})
    pq.write_table(table, output_path, compression="zstd", compression_level=3)
    print(f"Saved corpus to {output_path}")


def load_corpus(input_path: Path) -> Tuple[List[int], List[str]]:
    """Load saved corpus from parquet."""
    table = pq.read_table(input_path)
    return table.column("movieId").to_pylist(), table.column("text").to_pylist()


if __name__ == "__main__":
    config = load_config()
    movie_ids, corpus = load_cold_start_data(config)

    output_path = Path(config["parquet"]["output_dir"]).parent / "processed" / "cold_start_corpus.parquet"
    save_corpus(movie_ids, corpus, output_path)

    print(f"\nSample corpus entry:")
    print(f"  movieId: {movie_ids[0]}")
    print(f"  text: {corpus[0][:200]}...")