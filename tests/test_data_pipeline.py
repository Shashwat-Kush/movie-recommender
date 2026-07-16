"""Unit tests for build_aligned_metadata ordering and error handling."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cold_start import build_aligned_metadata


@pytest.fixture
def processed_dir(tmp_path):
    # Cold-start artifacts cover movieIds 10, 20, 30 (in that storage order)
    embeddings = np.arange(9, dtype=np.float32).reshape(3, 3)  # row i = [3i, 3i+1, 3i+2]
    np.save(tmp_path / "cold_start_embeddings_128.npy", embeddings)
    np.save(tmp_path / "cold_start_movie_ids.npy", np.array([10, 20, 30]))
    return tmp_path


def test_rows_land_in_movie_map_order(processed_dir):
    movie_map = {30: 0, 10: 1, 20: 2}  # deliberately different order than storage
    aligned = build_aligned_metadata(movie_map, processed_dir)
    assert aligned.shape == (3, 3)
    np.testing.assert_array_equal(aligned[0], [6, 7, 8])   # movieId 30 = storage row 2
    np.testing.assert_array_equal(aligned[1], [0, 1, 2])   # movieId 10 = storage row 0
    np.testing.assert_array_equal(aligned[2], [3, 4, 5])   # movieId 20 = storage row 1


def test_missing_movie_ids_zero_filled(processed_dir):
    movie_map = {10: 0, 99: 1}  # 99 has no cold-start row
    aligned = build_aligned_metadata(movie_map, processed_dir)
    np.testing.assert_array_equal(aligned[0], [0, 1, 2])
    np.testing.assert_array_equal(aligned[1], [0, 0, 0])


def test_length_mismatch_raises(processed_dir):
    np.save(processed_dir / "cold_start_movie_ids.npy", np.array([10, 20]))  # 2 ids, 3 rows
    with pytest.raises(ValueError, match="mismatch"):
        build_aligned_metadata({10: 0}, processed_dir)


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_aligned_metadata({10: 0}, tmp_path)
