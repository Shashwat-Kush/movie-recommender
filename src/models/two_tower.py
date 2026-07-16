"""Two-Tower retrieval model for MovieLens recommendations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """3-layer MLP with ReLU activations and optional dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTower(nn.Module):
    """Two-Tower model for retrieval.

    User Tower: 3-layer MLP (user_features_dim -> 256 -> 128)
    Item Tower: 3-layer MLP (item_features_dim -> 256 -> 128)

    Returns dot product of L2-normalized embeddings.
    """

    def __init__(
        self,
        user_features_dim: int,
        item_features_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_features_dim = user_features_dim
        self.item_features_dim = item_features_dim
        self.output_dim = output_dim

        self.user_tower = MLP(user_features_dim, hidden_dim, output_dim, dropout)
        self.item_tower = MLP(item_features_dim, hidden_dim, output_dim, dropout)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity scores.

        Args:
            user_features: (batch_size, user_features_dim)
            item_features: (batch_size, item_features_dim)

        Returns:
            scores: (batch_size,) dot product of L2-normalized embeddings
        """
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)

        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return (user_emb * item_emb).sum(dim=1)

    def get_user_embeddings(self, user_features: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized user embeddings."""
        emb = self.user_tower(user_features)
        return F.normalize(emb, p=2, dim=1)

    def get_item_embeddings(self, item_features: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized item embeddings."""
        emb = self.item_tower(item_features)
        return F.normalize(emb, p=2, dim=1)


class TwoTowerWithMetadata(nn.Module):
    """Two-Tower with metadata concatenation for item tower.

    User Tower: user_id embedding -> 3-layer MLP
    Item Tower: (item_id embedding + metadata) -> 3-layer MLP

    For cold-start, item_id embedding can be zero and metadata provides signal.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        metadata_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.metadata_dim = metadata_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        user_input_dim = embedding_dim
        item_input_dim = embedding_dim + metadata_dim

        self.user_tower = MLP(user_input_dim, hidden_dim, output_dim, dropout)
        self.item_tower = MLP(item_input_dim, hidden_dim, output_dim, dropout)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity scores.

        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size,)
            item_metadata: (batch_size, metadata_dim)

        Returns:
            scores: (batch_size,) dot product of L2-normalized embeddings
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        item_features = torch.cat([item_emb, item_metadata], dim=1)

        user_out = self.user_tower(user_emb)
        item_out = self.item_tower(item_features)

        user_out = F.normalize(user_out, p=2, dim=1)
        item_out = F.normalize(item_out, p=2, dim=1)

        return (user_out * item_out).sum(dim=1)

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized user embeddings from IDs."""
        emb = self.user_embedding(user_ids)
        emb = self.user_tower(emb)
        return F.normalize(emb, p=2, dim=1)

    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Get L2-normalized item embeddings from IDs and metadata."""
        item_emb = self.item_embedding(item_ids)
        item_features = torch.cat([item_emb, item_metadata], dim=1)
        emb = self.item_tower(item_features)
        return F.normalize(emb, p=2, dim=1)

    def get_item_embeddings_cold(self, item_metadata: torch.Tensor) -> torch.Tensor:
        """Get embeddings for cold-start items (no ID embedding)."""
        zero_emb = torch.zeros(item_metadata.size(0), self.embedding_dim, device=item_metadata.device)
        item_features = torch.cat([zero_emb, item_metadata], dim=1)
        emb = self.item_tower(item_features)
        return F.normalize(emb, p=2, dim=1)


class TwoTowerHistory(nn.Module):
    """Two-Tower whose user representation is the user's watch history, not an ID.

    The user vector is the mean of the item-embedding-table rows for the user's K
    most recent liked movies, passed through the user MLP. No per-user parameters:
    the model generalizes to any user with history and reacts to new watches without
    retraining. The item tower is identical to TwoTowerWithMetadata.

    The (n_users, K) history matrix (-1 padded) is a persistent buffer, so
    checkpoints are self-contained for evaluation and serving.
    """

    history_based = True  # trainer/eval pass exclude_items / rely on this flag

    def __init__(
        self,
        n_items: int,
        metadata_dim: int,
        history: torch.Tensor,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        id_dropout: float = 0.0,
        history_decay: float = 1.0,
    ):
        super().__init__()
        self.n_items = n_items
        self.metadata_dim = metadata_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        # Fraction of items whose ID embedding is zeroed during training so the item
        # tower learns to work from metadata alone — without it the cold path
        # (get_item_embeddings_cold) is out-of-distribution and useless (measured
        # 0.13% vs 10.56% warm recall@10).
        self.id_dropout = id_dropout
        # Recency weight for history pooling: position j (most recent first) gets
        # decay**j. 1.0 is a plain mean.
        self.history_decay = history_decay

        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        self.register_buffer("user_history", history.long())

        self.user_tower = MLP(embedding_dim, hidden_dim, output_dim, dropout)
        self.item_tower = MLP(embedding_dim + metadata_dim, hidden_dim, output_dim, dropout)

    def _pool_history(
        self,
        user_ids: torch.Tensor,
        exclude_items: torch.Tensor = None,
    ) -> torch.Tensor:
        """Recency-weighted mean of history item embeddings; -1 pads (and the
        excluded item) masked out.

        exclude_items drops the current training positive from its own user's history
        so the model can't answer by reading the label off its input.
        """
        hist = self.user_history[user_ids]  # (B, K)
        mask = hist >= 0
        if exclude_items is not None:
            mask &= hist != exclude_items.unsqueeze(1)

        weights = mask.float()
        if self.history_decay != 1.0:
            decay = self.history_decay ** torch.arange(hist.size(1), device=hist.device, dtype=torch.float32)
            weights = weights * decay

        emb = self.item_embedding(hist.clamp(min=0)) * weights.unsqueeze(-1)
        return emb.sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        exclude_items: torch.Tensor = None,
    ) -> torch.Tensor:
        """L2-normalized user embeddings pooled from watch history."""
        pooled = self._pool_history(user_ids, exclude_items)
        return F.normalize(self.user_tower(pooled), p=2, dim=1)

    def get_user_embedding_from_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """L2-normalized user embedding for an ad-hoc liked-movies list (cold-user
        onboarding — no user ID or stored history needed). item_ids: (K,) item
        indices ordered most-liked/most-recent first; same recency weighting as
        stored histories. Returns (1, output_dim)."""
        emb = self.item_embedding(item_ids)
        weights = torch.ones(len(item_ids), device=item_ids.device)
        if self.history_decay != 1.0:
            weights = self.history_decay ** torch.arange(len(item_ids), device=item_ids.device, dtype=torch.float32)
        pooled = (emb * weights.unsqueeze(1)).sum(dim=0, keepdim=True) / weights.sum().clamp(min=1e-8)
        return F.normalize(self.user_tower(pooled), p=2, dim=1)

    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """L2-normalized item embeddings from IDs and metadata.

        During training, id_dropout zeroes the ID embedding for a random fraction of
        the batch — exactly the cold-path input — so metadata-only items get real
        embeddings at serving time. Inactive under model.eval().
        """
        id_emb = self.item_embedding(item_ids)
        if self.training and self.id_dropout > 0:
            keep = (torch.rand(id_emb.size(0), 1, device=id_emb.device) >= self.id_dropout).float()
            id_emb = id_emb * keep
        item_features = torch.cat([id_emb, item_metadata], dim=1)
        return F.normalize(self.item_tower(item_features), p=2, dim=1)

    def get_item_embeddings_cold(self, item_metadata: torch.Tensor) -> torch.Tensor:
        """Get embeddings for cold-start items (no ID embedding)."""
        zero_emb = torch.zeros(item_metadata.size(0), self.embedding_dim, device=item_metadata.device)
        item_features = torch.cat([zero_emb, item_metadata], dim=1)
        return F.normalize(self.item_tower(item_features), p=2, dim=1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Similarity scores (dot of L2-normalized embeddings)."""
        user_out = self.get_user_embeddings(user_ids)
        item_out = self.get_item_embeddings(item_ids, item_metadata)
        return (user_out * item_out).sum(dim=1)