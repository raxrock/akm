"""Community detection in knowledge graphs."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A detected community of entities."""

    id: str
    entity_ids: List[str] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)
    centroid_id: Optional[str] = None
    modularity_score: float = 0.0
    label: Optional[str] = None


class CommunityDetector:
    """
    Detect communities in knowledge graphs.

    Implements Louvain algorithm for community detection.
    Falls back to simpler methods if NetworkX is not available.
    """

    def __init__(self) -> None:
        """Initialize the community detector."""
        self._use_networkx = False
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities

            self._use_networkx = True
            logger.info("Using NetworkX for community detection")
        except ImportError:
            logger.info("NetworkX not available, using fallback method")

    def detect(
        self,
        entities: Dict[str, str],
        edges: List[Tuple[int, int]],
        resolution: float = 1.0,
        min_community_size: int = 2,
    ) -> List[Community]:
        """
        Detect communities in the graph.

        Args:
            entities: Dict mapping entity_id to entity_name
            edges: List of (source_idx, target_idx) edges
            resolution: Resolution parameter for Louvain
            min_community_size: Minimum community size

        Returns:
            List of Community objects
        """
        if self._use_networkx:
            return self._detect_louvain(entities, edges, resolution, min_community_size)
        else:
            return self._detect_fallback(entities, edges, min_community_size)

    def _detect_louvain(
        self,
        entities: Dict[str, str],
        edges: List[Tuple[int, int]],
        resolution: float,
        min_community_size: int,
    ) -> List[Community]:
        """Detect communities using Louvain algorithm."""
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        # Build NetworkX graph
        G = nx.Graph()

        entity_list = list(entities.keys())
        for idx, entity_id in enumerate(entity_list):
            G.add_node(idx, entity_id=entity_id, name=entities[entity_id])

        for src, tgt in edges:
            if src < len(entity_list) and tgt < len(entity_list):
                G.add_edge(src, tgt)

        if G.number_of_nodes() == 0:
            return []

        # Detect communities
        try:
            communities_sets = louvain_communities(G, resolution=resolution)
        except Exception as e:
            logger.warning(f"Louvain detection failed: {e}")
            return self._detect_fallback(entities, edges, min_community_size)

        # Calculate modularity
        try:
            modularity = nx.algorithms.community.modularity(G, communities_sets)
        except Exception:
            modularity = 0.0

        # Convert to Community objects
        communities = []
        for community_set in communities_sets:
            if len(community_set) < min_community_size:
                continue

            entity_ids = [entity_list[idx] for idx in community_set if idx < len(entity_list)]
            entity_names = [entities[eid] for eid in entity_ids]

            # Find centroid (node with highest degree in community)
            centroid_id = None
            max_degree = -1
            subgraph = G.subgraph(community_set)
            for node in community_set:
                if node < len(entity_list):
                    degree = subgraph.degree(node)
                    if degree > max_degree:
                        max_degree = degree
                        centroid_id = entity_list[node]

            # Generate label from most common terms
            label = self._generate_label(entity_names)

            community = Community(
                id=str(uuid4()),
                entity_ids=entity_ids,
                entity_names=entity_names,
                centroid_id=centroid_id,
                modularity_score=modularity,
                label=label,
            )
            communities.append(community)

        return communities

    def _detect_fallback(
        self,
        entities: Dict[str, str],
        edges: List[Tuple[int, int]],
        min_community_size: int,
    ) -> List[Community]:
        """
        Fallback community detection using connected components.

        Simple approach that finds connected components as communities.
        """
        entity_list = list(entities.keys())

        # Build adjacency list
        adj: Dict[int, Set[int]] = defaultdict(set)
        for src, tgt in edges:
            if src < len(entity_list) and tgt < len(entity_list):
                adj[src].add(tgt)
                adj[tgt].add(src)

        # Find connected components using BFS
        visited = set()
        components = []

        for start in range(len(entity_list)):
            if start in visited:
                continue

            # BFS
            component = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)

                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= min_community_size:
                components.append(component)

        # Convert to Community objects
        communities = []
        for component in components:
            entity_ids = [entity_list[idx] for idx in component if idx < len(entity_list)]
            entity_names = [entities[eid] for eid in entity_ids]

            # Find centroid (most connected node)
            centroid_id = None
            max_connections = -1
            for idx in component:
                connections = len(adj[idx])
                if connections > max_connections:
                    max_connections = connections
                    centroid_id = entity_list[idx] if idx < len(entity_list) else None

            label = self._generate_label(entity_names)

            community = Community(
                id=str(uuid4()),
                entity_ids=entity_ids,
                entity_names=entity_names,
                centroid_id=centroid_id,
                modularity_score=0.0,  # Can't compute without full graph
                label=label,
            )
            communities.append(community)

        return communities

    def _generate_label(self, names: List[str], max_words: int = 3) -> str:
        """Generate a label for a community based on entity names."""
        if not names:
            return "Unknown"

        # Count word frequencies
        word_counts: Dict[str, int] = defaultdict(int)
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or"}

        for name in names:
            words = name.lower().replace("_", " ").split()
            for word in words:
                if len(word) > 2 and word not in stop_words:
                    word_counts[word] += 1

        if not word_counts:
            return names[0][:20] if names else "Unknown"

        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        label_words = [w[0].capitalize() for w in top_words[:max_words]]

        return " ".join(label_words)

    def get_community_summary(self, communities: List[Community]) -> Dict:
        """
        Get a summary of detected communities.

        Args:
            communities: List of detected communities

        Returns:
            Summary dictionary
        """
        if not communities:
            return {
                "num_communities": 0,
                "total_entities": 0,
                "avg_size": 0,
                "largest_community": None,
            }

        sizes = [len(c.entity_ids) for c in communities]
        largest = max(communities, key=lambda c: len(c.entity_ids))

        return {
            "num_communities": len(communities),
            "total_entities": sum(sizes),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "largest_community": {
                "id": largest.id,
                "label": largest.label,
                "size": len(largest.entity_ids),
            },
            "modularity": communities[0].modularity_score if communities else 0.0,
        }


__all__ = ["CommunityDetector", "Community"]
