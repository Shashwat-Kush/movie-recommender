"""Matrix Factorization model with ALS and SGD implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    """Matrix Factorization for explicit feedback (SGD) and implicit feedback (ALS)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        use_bias: bool = True,
        sparse: bool = False,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias

        self.user_embeddings = nn.Embedding(n_users, embedding_dim, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim, sparse=sparse)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        if use_bias:
            self.user_bias = nn.Embedding(n_users, 1, sparse=sparse)
            self.item_bias = nn.Embedding(n_items, 1, sparse=sparse)
            self.global_bias = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
        else:
            self.register_parameter("user_bias", None)
            self.register_parameter("item_bias", None)
            self.register_parameter("global_bias", None)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict ratings for user-item pairs."""
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        pred = (user_emb * item_emb).sum(dim=1)

        if self.use_bias:
            pred = pred + self.user_bias(user_ids).squeeze() + self.item_bias(item_ids).squeeze() + self.global_bias

        return pred

    def get_user_embeddings(self) -> torch.Tensor:
        return self.user_embeddings.weight.detach()

    def get_item_embeddings(self) -> torch.Tensor:
        return self.item_embeddings.weight.detach()


class ImplicitMF(nn.Module):
    """Matrix Factorization for implicit feedback using BPR loss."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        sparse: bool = False,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(n_users, embedding_dim, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim, sparse=sparse)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: torch.Tensor) -> torch.Tensor:
        """BPR loss: maximize score(pos) - score(neg)."""
        user_emb = self.user_embeddings(user_ids)
        pos_emb = self.item_embeddings(pos_item_ids)
        neg_emb = self.item_embeddings(neg_item_ids)

        pos_score = (user_emb * pos_emb).sum(dim=1)
        neg_score = (user_emb * neg_emb).sum(dim=1)

        return -F.logsigmoid(pos_score - neg_score).mean()

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict preference scores."""
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        return (user_emb * item_emb).sum(dim=1)

    def get_user_embeddings(self) -> torch.Tensor:
        return self.user_embeddings.weight.detach()

    def get_item_embeddings(self) -> torch.Tensor:
        return self.item_embeddings.weight.detach()


def als_update(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    ratings: torch.Tensor,
    reg: float = 0.01,
    n_factors: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single ALS update step (CPU, for reference/fallback)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    n_users = user_emb.shape[0]
    n_items = item_emb.shape[0]

    if ratings.is_sparse:
        ratings_coo = ratings.coalesce()
        rows = ratings_coo.indices()[0].cpu().numpy()
        cols = ratings_coo.indices()[1].cpu().numpy()
        vals = ratings_coo.values().cpu().numpy()
    else:
        rows, cols = ratings.nonzero()
        vals = ratings[rows, cols]
        rows = rows.cpu().numpy()
        cols = cols.cpu().numpy()
        vals = vals.cpu().numpy()

    R = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    for u in range(n_users):
        rated_items = R[u].indices
        if len(rated_items) == 0:
            continue
        R_u = R[u].toarray().flatten()
        V_rated = item_emb[rated_items].cpu().numpy()
        A = V_rated.T @ V_rated + reg * np.eye(n_factors)
        b = V_rated.T @ R_u[rated_items]
        user_emb[u] = torch.from_numpy(spsolve(A, b))

    for i in range(n_items):
        rated_users = R[:, i].indices
        if len(rated_users) == 0:
            continue
        R_i = R[:, i].toarray().flatten()
        U_rated = user_emb[rated_users].cpu().numpy()
        A = U_rated.T @ U_rated + reg * np.eye(n_factors)
        b = U_rated.T @ R_i[rated_users]
        item_emb[i] = torch.from_numpy(spsolve(A, b))

    return user_emb, item_emb