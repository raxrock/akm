"""Graph traversal with semantic context for query processing."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

from akm.core.interfaces import EmbeddingModel, GraphBackend
from akm.core.models import Entity, Link, Relationship, SearchResult, TraversalResult

logger = logging.getLogger(__name__)


@dataclass
class TraversalPath:
    """Represents a path through the graph."""

    nodes: List[Entity] = field(default_factory=list)
    edges: List[Relationship] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    total_weight: float = 0.0
    semantic_relevance: float = 0.0

    @property
    def length(self) -> int:
        """Get path length (number of edges)."""
        return len(self.edges) + len(self.links)

    def add_node(self, entity: Entity) -> None:
        """Add a node to the path."""
        self.nodes.append(entity)

    def add_edge(self, relationship: Relationship) -> None:
        """Add an edge to the path."""
        self.edges.append(relationship)
        self.total_weight += relationship.confidence

    def add_link(self, link: Link) -> None:
        """Add a soft link to the path."""
        self.links.append(link)
        self.total_weight += link.weight.value


@dataclass
class TraversalConfig:
    """Configuration for graph traversal."""

    max_depth: int = 3
    max_paths: int = 10
    min_edge_weight: float = 0.1
    min_link_weight: float = 0.3
    include_links: bool = True
    semantic_filter: bool = True
    semantic_threshold: float = 0.3
    relationship_types: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    prioritize_validated_links: bool = True


class SemanticTraverser:
    """
    Graph traverser that considers semantic context.

    This traverser:
    1. Uses semantic similarity to guide path exploration
    2. Considers both relationships and adaptive links
    3. Weights paths by relevance to a query
    4. Supports various traversal strategies
    """

    def __init__(
        self,
        graph: GraphBackend,
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[TraversalConfig] = None,
    ) -> None:
        """
        Initialize the semantic traverser.

        Args:
            graph: Graph backend for entity/relationship access
            embedding_model: Optional embedding model for semantic similarity
            config: Traversal configuration
        """
        self._graph = graph
        self._embedding = embedding_model
        self.config = config or TraversalConfig()

    def traverse_from_entity(
        self,
        start_id: str,
        query_embedding: Optional[List[float]] = None,
        links: Optional[Dict[str, List[Link]]] = None,
    ) -> TraversalResult:
        """
        Traverse from a starting entity.

        Args:
            start_id: Starting entity ID
            query_embedding: Optional query embedding for semantic filtering
            links: Optional map of entity_id -> links

        Returns:
            Traversal result with discovered entities and paths
        """
        start_entity = self._graph.get_entity(start_id)
        if not start_entity:
            return TraversalResult(
                start_entity=Entity(name="Unknown", id=UUID(start_id)),
                entities=[],
                relationships=[],
                links=[],
                paths=[],
                depth_reached=0,
            )

        visited: Set[str] = {start_id}
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        discovered_links: List[Link] = []
        paths: List[List[UUID]] = []

        # BFS traversal
        current_level = [(start_entity, [UUID(start_id)])]
        depth = 0

        while current_level and depth < self.config.max_depth:
            next_level = []
            depth += 1

            for entity, path in current_level:
                # Get relationships
                entity_rels = self._graph.get_relationships(
                    str(entity.id),
                    direction="both",
                    relationship_type=self._get_rel_type_filter(),
                )

                for rel in entity_rels:
                    # Filter by weight
                    if rel.confidence < self.config.min_edge_weight:
                        continue

                    # Get target entity
                    target_id = str(rel.target_id) if str(rel.source_id) == str(entity.id) else str(rel.source_id)

                    if target_id in visited:
                        continue

                    target = self._graph.get_entity(target_id)
                    if not target:
                        continue

                    # Apply entity type filter
                    if not self._passes_entity_filter(target):
                        continue

                    # Apply semantic filter
                    if query_embedding and self.config.semantic_filter:
                        if not self._passes_semantic_filter(target, query_embedding):
                            continue

                    visited.add(target_id)
                    entities.append(target)
                    relationships.append(rel)

                    new_path = path + [UUID(target_id)]
                    paths.append(new_path)

                    if depth < self.config.max_depth:
                        next_level.append((target, new_path))

                # Process adaptive links if enabled
                if self.config.include_links and links:
                    entity_links = links.get(str(entity.id), [])
                    for link in entity_links:
                        if link.weight.value < self.config.min_link_weight:
                            continue

                        target_id = str(link.target_id) if str(link.source_id) == str(entity.id) else str(link.source_id)

                        if target_id in visited:
                            continue

                        target = self._graph.get_entity(target_id)
                        if not target:
                            continue

                        if not self._passes_entity_filter(target):
                            continue

                        if query_embedding and self.config.semantic_filter:
                            if not self._passes_semantic_filter(target, query_embedding):
                                continue

                        visited.add(target_id)
                        entities.append(target)
                        discovered_links.append(link)

                        new_path = path + [UUID(target_id)]
                        paths.append(new_path)

                        if depth < self.config.max_depth:
                            next_level.append((target, new_path))

            current_level = next_level

        return TraversalResult(
            start_entity=start_entity,
            entities=entities,
            relationships=relationships,
            links=discovered_links,
            paths=paths[:self.config.max_paths],
            depth_reached=depth,
        )

    def find_paths_between(
        self,
        source_id: str,
        target_id: str,
        links: Optional[Dict[str, List[Link]]] = None,
    ) -> List[TraversalPath]:
        """
        Find all paths between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            links: Optional map of entity_id -> links

        Returns:
            List of paths between the entities
        """
        source = self._graph.get_entity(source_id)
        target = self._graph.get_entity(target_id)

        if not source or not target:
            return []

        paths: List[TraversalPath] = []
        self._dfs_find_paths(
            source,
            target_id,
            TraversalPath(nodes=[source]),
            set([source_id]),
            links or {},
            paths,
            0,
        )

        # Sort by total weight (higher is better)
        paths.sort(key=lambda p: p.total_weight, reverse=True)
        return paths[:self.config.max_paths]

    def _dfs_find_paths(
        self,
        current: Entity,
        target_id: str,
        current_path: TraversalPath,
        visited: Set[str],
        links: Dict[str, List[Link]],
        found_paths: List[TraversalPath],
        depth: int,
    ) -> None:
        """DFS helper to find paths."""
        if depth > self.config.max_depth:
            return

        if str(current.id) == target_id:
            found_paths.append(current_path)
            return

        # Explore relationships
        rels = self._graph.get_relationships(
            str(current.id),
            direction="both",
        )

        for rel in rels:
            if rel.confidence < self.config.min_edge_weight:
                continue

            next_id = str(rel.target_id) if str(rel.source_id) == str(current.id) else str(rel.source_id)

            if next_id in visited:
                continue

            next_entity = self._graph.get_entity(next_id)
            if not next_entity:
                continue

            # Create new path
            new_path = TraversalPath(
                nodes=current_path.nodes + [next_entity],
                edges=current_path.edges + [rel],
                links=list(current_path.links),
                total_weight=current_path.total_weight + rel.confidence,
            )

            self._dfs_find_paths(
                next_entity,
                target_id,
                new_path,
                visited | {next_id},
                links,
                found_paths,
                depth + 1,
            )

        # Explore links
        if self.config.include_links:
            entity_links = links.get(str(current.id), [])
            for link in entity_links:
                if link.weight.value < self.config.min_link_weight:
                    continue

                next_id = str(link.target_id) if str(link.source_id) == str(current.id) else str(link.source_id)

                if next_id in visited:
                    continue

                next_entity = self._graph.get_entity(next_id)
                if not next_entity:
                    continue

                new_path = TraversalPath(
                    nodes=current_path.nodes + [next_entity],
                    edges=list(current_path.edges),
                    links=current_path.links + [link],
                    total_weight=current_path.total_weight + link.weight.value,
                )

                self._dfs_find_paths(
                    next_entity,
                    target_id,
                    new_path,
                    visited | {next_id},
                    links,
                    found_paths,
                    depth + 1,
                )

    def expand_context(
        self,
        seed_entities: List[Entity],
        query_embedding: Optional[List[float]] = None,
        links: Optional[Dict[str, List[Link]]] = None,
    ) -> Tuple[List[Entity], List[Relationship], List[Link]]:
        """
        Expand context from seed entities.

        Args:
            seed_entities: Starting entities
            query_embedding: Optional query embedding for filtering
            links: Optional map of entity_id -> links

        Returns:
            Tuple of (entities, relationships, links)
        """
        all_entities: Dict[str, Entity] = {str(e.id): e for e in seed_entities}
        all_relationships: List[Relationship] = []
        all_links: List[Link] = []
        seen_rels: Set[str] = set()
        seen_links: Set[str] = set()

        for seed in seed_entities:
            result = self.traverse_from_entity(
                str(seed.id),
                query_embedding=query_embedding,
                links=links,
            )

            for entity in result.entities:
                if str(entity.id) not in all_entities:
                    all_entities[str(entity.id)] = entity

            for rel in result.relationships:
                if str(rel.id) not in seen_rels:
                    all_relationships.append(rel)
                    seen_rels.add(str(rel.id))

            for link in result.links:
                if str(link.id) not in seen_links:
                    all_links.append(link)
                    seen_links.add(str(link.id))

        return list(all_entities.values()), all_relationships, all_links

    def rank_entities_by_relevance(
        self,
        entities: List[Entity],
        query_embedding: List[float],
    ) -> List[Tuple[Entity, float]]:
        """
        Rank entities by semantic relevance to query.

        Args:
            entities: Entities to rank
            query_embedding: Query embedding

        Returns:
            List of (entity, score) tuples sorted by relevance
        """
        if not self._embedding:
            return [(e, 1.0) for e in entities]

        scored_entities = []
        for entity in entities:
            if entity.embedding:
                score = self._compute_similarity(query_embedding, entity.embedding)
            else:
                # Generate embedding on the fly
                text = f"{entity.name}. {entity.description or ''}"
                entity_embedding = self._embedding.embed_single(text)
                score = self._compute_similarity(query_embedding, entity_embedding)

            scored_entities.append((entity, score))

        scored_entities.sort(key=lambda x: x[1], reverse=True)
        return scored_entities

    def _passes_entity_filter(self, entity: Entity) -> bool:
        """Check if entity passes type filter."""
        if not self.config.entity_types:
            return True

        entity_type = entity.entity_type
        if hasattr(entity_type, "value"):
            entity_type = entity_type.value

        return entity_type in self.config.entity_types

    def _passes_semantic_filter(
        self,
        entity: Entity,
        query_embedding: List[float],
    ) -> bool:
        """Check if entity passes semantic similarity filter."""
        if not entity.embedding:
            return True  # No embedding means we can't filter

        similarity = self._compute_similarity(query_embedding, entity.embedding)
        return similarity >= self.config.semantic_threshold

    def _get_rel_type_filter(self) -> Optional[str]:
        """Get relationship type filter for queries."""
        if not self.config.relationship_types:
            return None
        if len(self.config.relationship_types) == 1:
            return self.config.relationship_types[0]
        return None  # Multiple types need different handling

    def _compute_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity between vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class ContextualTraverser:
    """
    Traverser that builds semantic context around entities.

    This is useful for:
    1. Building context for LLM queries
    2. Finding related information for answers
    3. Understanding entity neighborhoods
    """

    def __init__(
        self,
        graph: GraphBackend,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        """
        Initialize contextual traverser.

        Args:
            graph: Graph backend
            embedding_model: Optional embedding model
        """
        self._graph = graph
        self._embedding = embedding_model

    def get_entity_context(
        self,
        entity_id: str,
        max_items: int = 10,
    ) -> Dict[str, Any]:
        """
        Get contextual information for an entity.

        Args:
            entity_id: Entity ID
            max_items: Maximum related items per category

        Returns:
            Context dictionary with related entities, relationships, etc.
        """
        entity = self._graph.get_entity(entity_id)
        if not entity:
            return {"entity": None, "context": {}}

        context = {
            "direct_relationships": [],
            "neighbors": [],
            "related_by_type": [],
        }

        # Get direct relationships
        relationships = self._graph.get_relationships(entity_id, direction="both")
        context["direct_relationships"] = [
            {
                "type": rel.relationship_type,
                "target_id": str(rel.target_id) if str(rel.source_id) == entity_id else str(rel.source_id),
                "confidence": rel.confidence,
            }
            for rel in relationships[:max_items]
        ]

        # Get neighbors
        neighbors = self._graph.get_neighbors(entity_id, depth=1)
        context["neighbors"] = [
            {
                "id": str(n.id),
                "name": n.name,
                "type": n.entity_type.value if hasattr(n.entity_type, "value") else n.entity_type,
            }
            for n in neighbors[:max_items]
        ]

        # Get related by type
        entity_type = entity.entity_type.value if hasattr(entity.entity_type, "value") else entity.entity_type
        related = self._graph.find_entities(entity_type=entity_type, limit=max_items + 1)
        context["related_by_type"] = [
            {
                "id": str(r.id),
                "name": r.name,
            }
            for r in related
            if str(r.id) != entity_id
        ][:max_items]

        return {
            "entity": {
                "id": str(entity.id),
                "name": entity.name,
                "type": entity_type,
                "description": entity.description,
                "properties": entity.properties,
            },
            "context": context,
        }

    def build_subgraph(
        self,
        entity_ids: List[str],
        include_relationships: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a subgraph from a set of entities.

        Args:
            entity_ids: Entity IDs to include
            include_relationships: Whether to include relationships between entities

        Returns:
            Subgraph with nodes and edges
        """
        nodes = []
        edges = []
        entity_set = set(entity_ids)

        for entity_id in entity_ids:
            entity = self._graph.get_entity(entity_id)
            if entity:
                nodes.append({
                    "id": str(entity.id),
                    "name": entity.name,
                    "type": entity.entity_type.value if hasattr(entity.entity_type, "value") else entity.entity_type,
                    "properties": entity.properties,
                })

        if include_relationships:
            seen_edges = set()
            for entity_id in entity_ids:
                relationships = self._graph.get_relationships(entity_id, direction="outgoing")
                for rel in relationships:
                    if str(rel.target_id) in entity_set:
                        edge_key = f"{rel.source_id}-{rel.target_id}-{rel.relationship_type}"
                        if edge_key not in seen_edges:
                            edges.append({
                                "source": str(rel.source_id),
                                "target": str(rel.target_id),
                                "type": rel.relationship_type,
                                "confidence": rel.confidence,
                            })
                            seen_edges.add(edge_key)

        return {
            "nodes": nodes,
            "edges": edges,
        }
