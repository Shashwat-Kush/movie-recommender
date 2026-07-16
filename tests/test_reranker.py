"""Unit tests for LLMReranker._parse_response (no network, no API key)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.reranker import LLMReranker


@pytest.fixture
def reranker():
    return LLMReranker(api_key="test-key-not-used", cache_dir=None)


def test_json_object_ranking(reranker):
    results = reranker._parse_response('{"ranking": [4, 0, 7]}', num_candidates=10)
    assert [r["index"] for r in results] == [4, 0, 7]
    # synthesized scores strictly decrease with rank
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bare_list(reranker):
    results = reranker._parse_response("[2, 1]", num_candidates=5)
    assert [r["index"] for r in results] == [2, 1]


def test_salvage_from_prose_wrapped_output(reranker):
    text = 'Here is my ranking:\n[3, 1, 4'  # truncated, invalid JSON
    results = reranker._parse_response(text, num_candidates=5)
    assert [r["index"] for r in results] == [3, 1, 4]


def test_garbage_returns_empty(reranker):
    assert reranker._parse_response("no json here at all", num_candidates=5) == []
    assert reranker._parse_response("", num_candidates=5) == []


def test_out_of_range_and_duplicate_indices_dropped(reranker):
    results = reranker._parse_response('{"ranking": [1, 9, 1, -2, 0]}', num_candidates=3)
    assert [r["index"] for r in results] == [1, 0]


def test_dict_with_unexpected_key_still_parses(reranker):
    results = reranker._parse_response('{"order": [0, 2]}', num_candidates=3)
    assert [r["index"] for r in results] == [0, 2]


def test_rerank_pads_short_llm_output_with_retrieval_order(reranker, monkeypatch):
    movies = [{"movieId": m, "title": f"t{m}", "genres": ""} for m in range(6)]
    # LLM only ranks two candidates; the rest must be padded in retrieval order
    monkeypatch.setattr(
        reranker, "_rerank_batch",
        lambda query, batch, top_k: [{"index": 3, "score": 2.0}, {"index": 1, "score": 1.0}],
    )
    out = reranker.rerank("q", movies, top_k=5)
    assert [m["movieId"] for m in out] == [3, 1, 0, 2, 4]


def test_rerank_full_fallback_keeps_retrieval_order(reranker, monkeypatch):
    movies = [{"movieId": m, "title": f"t{m}", "genres": ""} for m in range(4)]
    monkeypatch.setattr(reranker, "_rerank_batch", lambda query, batch, top_k: [])
    out = reranker.rerank("q", movies, top_k=3)
    assert [m["movieId"] for m in out] == [0, 1, 2]
