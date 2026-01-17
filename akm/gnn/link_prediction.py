"""Link prediction using Graph Neural Networks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GNNTrainingResult:
    """Result from GNN training."""

    model_type: str
    epochs: int
    final_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0


class LinkPredictor:
    """
    Link prediction using Graph Neural Networks.

    Uses a simple GNN architecture (GraphSAGE-like) for learning
    node embeddings and predicting links.

    If PyTorch Geometric is not available, falls back to a simpler
    matrix factorization approach.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_channels: int = 64,
        learning_rate: float = 0.01,
        embedding_dim: int = 32,
    ) -> None:
        """
        Initialize the link predictor.

        Args:
            num_nodes: Number of nodes in the graph
            hidden_channels: Hidden layer dimension
            learning_rate: Learning rate for training
            embedding_dim: Final embedding dimension
        """
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        self._model: Optional[Any] = None
        self._embeddings: Optional[np.ndarray] = None
        self._use_pytorch = False

        # Try to import PyTorch Geometric
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import SAGEConv

            self._use_pytorch = True
            logger.info("Using PyTorch Geometric for link prediction")
        except ImportError:
            logger.info("PyTorch Geometric not available, using fallback method")

    def train(
        self,
        edges: List[Tuple[int, int]],
        epochs: int = 100,
    ) -> GNNTrainingResult:
        """
        Train the link prediction model.

        Args:
            edges: List of (source_idx, target_idx) edges
            epochs: Number of training epochs

        Returns:
            GNNTrainingResult with training metrics
        """
        start_time = time.time()

        if self._use_pytorch:
            result = self._train_pytorch(edges, epochs)
        else:
            result = self._train_fallback(edges, epochs)

        result.training_time_seconds = time.time() - start_time
        return result

    def predict(
        self,
        num_nodes: int,
        existing_edges: List[Tuple[int, int]],
        top_k: int = 10,
        min_probability: float = 0.5,
    ) -> List[Tuple[int, int, float]]:
        """
        Predict new links.

        Args:
            num_nodes: Total number of nodes
            existing_edges: Existing edges to exclude
            top_k: Number of predictions to return
            min_probability: Minimum probability threshold

        Returns:
            List of (source_idx, target_idx, probability) tuples
        """
        if self._embeddings is None:
            return []

        # Convert existing edges to set for fast lookup
        existing = set()
        for src, tgt in existing_edges:
            existing.add((src, tgt))
            existing.add((tgt, src))  # Undirected

        predictions = []

        # Compute scores for all non-existing pairs
        for i in range(min(num_nodes, len(self._embeddings))):
            for j in range(i + 1, min(num_nodes, len(self._embeddings))):
                if (i, j) in existing or (j, i) in existing:
                    continue

                # Compute similarity score
                emb_i = self._embeddings[i]
                emb_j = self._embeddings[j]

                # Dot product similarity
                score = np.dot(emb_i, emb_j)
                # Convert to probability using sigmoid
                prob = 1 / (1 + np.exp(-score))

                if prob >= min_probability:
                    predictions.append((i, j, float(prob)))

        # Sort by probability and return top_k
        predictions.sort(key=lambda x: x[2], reverse=True)
        return predictions[:top_k]

    def get_embeddings(self) -> np.ndarray:
        """Get learned node embeddings."""
        if self._embeddings is None:
            return np.array([])
        return self._embeddings

    def _train_pytorch(
        self,
        edges: List[Tuple[int, int]],
        epochs: int,
    ) -> GNNTrainingResult:
        """Train using PyTorch Geometric."""
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv

        # Build edge index
        edge_index = torch.tensor(
            [[e[0] for e in edges] + [e[1] for e in edges],
             [e[1] for e in edges] + [e[0] for e in edges]],
            dtype=torch.long
        )

        # Initial node features (random or one-hot)
        x = torch.randn(self.num_nodes, self.hidden_channels)

        # Define simple GNN model
        class GNNModel(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                return x

            def decode(self, z, edge_index):
                return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        model = GNNModel(self.hidden_channels, self.hidden_channels, self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training loop
        model.train()
        final_loss = 0.0
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Get embeddings
            z = model(x, edge_index)

            # Positive edges
            pos_edge_index = edge_index
            pos_out = model.decode(z, pos_edge_index)

            # Negative sampling
            neg_edge_index = self._negative_sampling(edge_index, self.num_nodes)
            neg_out = model.decode(z, neg_edge_index)

            # Binary cross entropy loss
            pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()

            final_loss = loss.item()
            losses.append(final_loss)

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: loss={final_loss:.4f}")

        # Store model and embeddings
        model.eval()
        with torch.no_grad():
            self._embeddings = model(x, edge_index).numpy()
        self._model = model

        return GNNTrainingResult(
            model_type="pytorch_sage",
            epochs=epochs,
            final_loss=final_loss,
            metrics={
                "avg_loss": np.mean(losses),
                "min_loss": np.min(losses),
            },
        )

    def _train_fallback(
        self,
        edges: List[Tuple[int, int]],
        epochs: int,
    ) -> GNNTrainingResult:
        """
        Fallback training using matrix factorization.

        When PyTorch Geometric is not available, use a simple
        matrix factorization approach to learn embeddings.
        """
        logger.info("Using matrix factorization fallback")

        # Initialize embeddings randomly
        self._embeddings = np.random.randn(
            self.num_nodes, self.embedding_dim
        ).astype(np.float32)

        # Build adjacency matrix
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for src, tgt in edges:
            if src < self.num_nodes and tgt < self.num_nodes:
                adj[src, tgt] = 1.0
                adj[tgt, src] = 1.0

        # Simple gradient descent on embedding similarity
        final_loss = 0.0
        losses = []

        for epoch in range(epochs):
            # Compute predictions
            pred = self._embeddings @ self._embeddings.T

            # Compute loss (MSE between adjacency and prediction)
            loss = np.mean((adj - pred) ** 2)

            # Gradient
            grad = -2 * (adj - pred) @ self._embeddings / self.num_nodes

            # Update embeddings
            self._embeddings -= self.learning_rate * grad

            # Normalize embeddings
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            self._embeddings = self._embeddings / (norms + 1e-8)

            final_loss = loss
            losses.append(loss)

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: loss={loss:.4f}")

        return GNNTrainingResult(
            model_type="matrix_factorization",
            epochs=epochs,
            final_loss=float(final_loss),
            metrics={
                "avg_loss": float(np.mean(losses)),
                "min_loss": float(np.min(losses)),
            },
        )

    def _negative_sampling(
        self,
        edge_index: Any,
        num_nodes: int,
        num_neg_samples: Optional[int] = None,
    ) -> Any:
        """Generate negative edge samples."""
        import torch

        num_edges = edge_index.size(1) // 2  # Divide by 2 for undirected
        num_neg = num_neg_samples or num_edges

        # Random negative edges
        neg_src = torch.randint(0, num_nodes, (num_neg,))
        neg_dst = torch.randint(0, num_nodes, (num_neg,))

        # Make sure they're different nodes
        mask = neg_src != neg_dst
        neg_src = neg_src[mask]
        neg_dst = neg_dst[mask]

        return torch.stack([neg_src, neg_dst], dim=0)


__all__ = ["LinkPredictor", "GNNTrainingResult"]
