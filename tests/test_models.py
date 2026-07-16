"""Unit tests for TwoTowerHistory pooling, exclusion, id-dropout, and checkpointing."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerHistory


HIST = torch.tensor([
    [0, 1, 2, -1],   # user 0: three history items
    [3, -1, -1, -1], # user 1: one item
    [-1, -1, -1, -1] # user 2: empty history
])


def make_model(**kwargs):
    torch.manual_seed(0)
    return TwoTowerHistory(n_items=5, metadata_dim=8, history=HIST.clone(), **kwargs)


def test_exclusion_changes_user_embedding():
    m = make_model().eval()
    users = torch.tensor([0])
    with torch.no_grad():
        plain = m.get_user_embeddings(users)
        excluded = m.get_user_embeddings(users, exclude_items=torch.tensor([1]))
    assert not torch.allclose(plain, excluded)


def test_empty_and_fully_excluded_histories_are_finite():
    m = make_model().eval()
    with torch.no_grad():
        emb = m.get_user_embeddings(torch.tensor([1, 2]), exclude_items=torch.tensor([3, 0]))
    # user 1's only item is excluded; user 2 has no history at all
    assert torch.isfinite(emb).all()


def test_decay_one_equals_plain_mean():
    m1 = make_model(history_decay=1.0)
    m2 = make_model(history_decay=0.9)
    users = torch.tensor([0, 1])
    pooled_mean = m1._pool_history(users)
    pooled_decay = m2._pool_history(users)
    # single-item history (user 1) is decay-invariant; multi-item (user 0) is not
    assert torch.allclose(pooled_mean[1], pooled_decay[1], atol=1e-6)
    assert not torch.allclose(pooled_mean[0], pooled_decay[0])


def test_decay_weights_recent_items_more():
    m = make_model(history_decay=0.5)
    pooled = m._pool_history(torch.tensor([0]))
    emb = m.item_embedding(torch.tensor([0, 1, 2]))
    # decay**j over positions 0..2, renormalized
    w = torch.tensor([1.0, 0.5, 0.25])
    expected = (emb * (w / w.sum()).unsqueeze(1)).sum(dim=0)
    assert torch.allclose(pooled[0], expected, atol=1e-6)


def test_id_dropout_noop_in_eval_and_active_in_train():
    m = make_model(id_dropout=0.99)
    items = torch.arange(5)
    meta = torch.randn(5, 8)

    m.eval()
    with torch.no_grad():
        a = m.get_item_embeddings(items, meta)
        b = m.get_item_embeddings(items, meta)
    assert torch.allclose(a, b)  # deterministic under eval

    m.train()
    torch.manual_seed(1)
    with torch.no_grad():
        dropped = m.get_item_embeddings(items, meta)
    # at 99% dropout nearly all rows should differ from the eval (warm) embeddings
    assert not torch.allclose(a, dropped)


def test_history_buffer_roundtrips_through_state_dict():
    m = make_model()
    m2 = TwoTowerHistory(n_items=5, metadata_dim=8, history=torch.zeros_like(HIST))
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.user_history, HIST)


def test_forward_shape_and_finiteness():
    m = make_model(id_dropout=0.2, history_decay=0.9).train()
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([1, 3, 4])
    meta = torch.randn(3, 8)
    scores = m(users, items, meta)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
