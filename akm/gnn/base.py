"""Base GNN manager for graph neural network operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from akm.core.config import GNNConfig
from akm.core.interfaces import GraphBackend
from akm.core.models import Entity, Link

logger = logging.getLogger(__name__)


@dataclass
class LinkPrediction:
    """A predicted link between two entities."""

    source_id: str
    target_id: str
    probability: float
    source_name: str = ""
    target_name: str = ""


@dataclass
class Community:
    """A detected community of entities."""

    id: str
    entity_ids: List[str] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)
    centroid_id: Optional[str] = None
    modularity_score: float = 0.0
    label: Optional[str] = None


@dataclass
class GNNTrainingResult:
    """Result from GNN training."""

    model_type: str
    epochs: int
    final_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0


class GNNManager:
    """
    Manager for Graph Neural Network operations.

    Provides:
    - Link prediction (predict new relationships)
    - Community detection (find clusters)
    - Node embeddings (learn representations)

    Example usage:
        gnn = GNNManager(graph=graph_backend, config=gnn_config)

        # Train link prediction model
        result = gnn.train_link_prediction(epochs=100)

        # Predict new links
        predictions = gnn.predict_links(top_k=10)

        # Detect communities
        communities = gnn.detect_communities()
    """

    def __init__(
        self,
        graph: GraphBackend,
        config: Optional[GNNConfig] = None,
    ) -> None:
        """
        Initialize the GNN manager.

        Args:
            graph: Graph backend for data access
            config: Optional GNN configuration
        """
        self.graph = graph
        self.config = config or GNNConfig()

        self._link_predictor: Optional[Any] = None
        self._community_detector: Optional[Any] = None
        self._node_embeddings: Optional[Dict[str, List[float]]] = None

    def train_link_prediction(
        self,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        hidden_channels: Optional[int] = None,
    ) -> GNNTrainingResult:
        """
        Train the link prediction model.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            hidden_channels: Hidden layer size

        Returns:
            GNNTrainingResult with training metrics
        """
        from akm.gnn.link_prediction import LinkPredictor

        epochs = epochs or self.config.epochs
        learning_rate = learning_rate or self.config.learning_rate
        hidden_channels = hidden_channels or self.config.hidden_channels

        # Get graph data
        entities, edges = self._get_graph_data()

        if len(entities) < 2:
            logger.warning("Not enough entities for link prediction")
            return GNNTrainingResult(
                model_type="link_prediction",
                epochs=0,
                final_loss=0.0,
                metrics={"error": "insufficient_data"},
            )

        # Initialize and train predictor
        self._link_predictor = LinkPredictor(
            num_nodes=len(entities),
            hidden_channels=hidden_channels,
            learning_rate=learning_rate,
        )

        result = self._link_predictor.train(
            edges=edges,
            epochs=epochs,
        )

        logger.info(f"Link prediction training complete: loss={result.final_loss:.4f}")
        return result

    def predict_links(
        self,
        top_k: int = 10,
        min_probability: float = 0.5,
        exclude_existing: bool = True,
    ) -> List[LinkPrediction]:
        """
        Predict new links between entities.

        Args:
            top_k: Number of top predictions to return
            min_probability: Minimum probability threshold
            exclude_existing: Whether to exclude existing relationships

        Returns:
            List of LinkPrediction objects
        """
        if self._link_predictor is None:
            logger.warning("Link predictor not trained. Call train_link_prediction() first.")
            return []

        # Get graph data
        entities, edges = self._get_graph_data()
        entity_list = list(entities.keys())

        # Get predictions
        predictions = self._link_predictor.predict(
            num_nodes=len(entity_list),
            existing_edges=edges if exclude_existing else [],
            top_k=top_k,
            min_probability=min_probability,
        )

        # Map back to entity IDs and names
        results = []
        for src_idx, tgt_idx, prob in predictions:
            if src_idx < len(entity_list) and tgt_idx < len(entity_list):
                src_id = entity_list[src_idx]
                tgt_id = entity_list[tgt_idx]
                src_name = entities[src_id]
                tgt_name = entities[tgt_id]

                results.append(LinkPrediction(
                    source_id=src_id,
                    target_id=tgt_id,
                    probability=prob,
                    source_name=src_name,
                    target_name=tgt_name,
                ))

        return results

    def detect_communities(
        self,
        resolution: float = 1.0,
        min_community_size: int = 2,
    ) -> List[Community]:
        """
        Detect communities in the graph.

        Uses Louvain algorithm for community detection.

        Args:
            resolution: Resolution parameter (higher = more communities)
            min_community_size: Minimum community size to include

        Returns:
            List of Community objects
        """
        from akm.gnn.community_detection import CommunityDetector

        # Get graph data
        entities, edges = self._get_graph_data()

        if len(entities) < 2:
            return []

        detector = CommunityDetector()
        communities = detector.detect(
            entities=entities,
            edges=edges,
            resolution=resolution,
            min_community_size=min_community_size,
        )

        return communities

    def get_node_embeddings(
        self,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """
        Get learned node embeddings.

        Args:
            entity_ids: Optional list of entity IDs to get embeddings for

        Returns:
            Dictionary mapping entity ID to embedding vector
        """
        if self._link_predictor is None:
            logger.warning("No model trained. Call train_link_prediction() first.")
            return {}

        entities, _ = self._get_graph_data()
        entity_list = list(entities.keys())

        embeddings = self._link_predictor.get_embeddings()

        result = {}
        for idx, emb in enumerate(embeddings):
            if idx < len(entity_list):
                entity_id = entity_list[idx]
                if entity_ids is None or entity_id in entity_ids:
                    result[entity_id] = emb.tolist()

        return result

    def find_similar_entities(
        self,
        entity_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find entities similar to a given entity based on embeddings.

        Args:
            entity_id: Entity ID to find similar entities for
            top_k: Number of similar entities to return

        Returns:
            List of (entity_id, similarity_score) tuples
        """
        embeddings = self.get_node_embeddings()

        if entity_id not in embeddings:
            return []

        import numpy as np

        target_emb = np.array(embeddings[entity_id])

        similarities = []
        for other_id, other_emb in embeddings.items():
            if other_id != entity_id:
                other_emb = np.array(other_emb)
                # Cosine similarity
                sim = np.dot(target_emb, other_emb) / (
                    np.linalg.norm(target_emb) * np.linalg.norm(other_emb) + 1e-8
                )
                similarities.append((other_id, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _get_graph_data(self) -> Tuple[Dict[str, str], List[Tuple[int, int]]]:
        """
        Get graph data in format suitable for GNN.

        Returns:
            Tuple of:
            - entities: Dict mapping entity_id to entity_name
            - edges: List of (source_idx, target_idx) tuples
        """
        # Get all entities
        all_entities = self.graph.find_entities(limit=10000)
        entities = {str(e.id): e.name for e in all_entities}

        # Create index mapping
        entity_to_idx = {eid: idx for idx, eid in enumerate(entities.keys())}

        # Get all relationships as edges
        edges = []
        for entity in all_entities:
            relationships = self.graph.get_relationships(str(entity.id))
            for rel in relationships:
                src_idx = entity_to_idx.get(rel.source_id)
                tgt_idx = entity_to_idx.get(rel.target_id)
                if src_idx is not None and tgt_idx is not None:
                    edges.append((src_idx, tgt_idx))

        return entities, edges


__all__ = ["GNNManager", "LinkPrediction", "Community", "GNNTrainingResult"]
