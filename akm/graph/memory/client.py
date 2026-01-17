"""In-memory graph backend implementation for testing."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

import networkx as nx

from akm.core.config import MemoryGraphConfig
from akm.core.exceptions import EntityNotFoundError
from akm.core.interfaces import GraphBackend
from akm.core.models import Entity, Relationship, TraversalResult

logger = logging.getLogger(__name__)


class MemoryGraphBackend(GraphBackend[Entity, Relationship]):
    """In-memory implementation of the graph backend using NetworkX."""

    def __init__(self, config: Optional[MemoryGraphConfig] = None) -> None:
        """
        Initialize in-memory graph backend.

        Args:
            config: Optional configuration for persistence
        """
        self._config = config or MemoryGraphConfig()
        self._graph: nx.DiGraph = nx.DiGraph()
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._connected = False

    def connect(self) -> None:
        """Initialize the graph (optionally load from disk)."""
        self._graph = nx.DiGraph()
        self._entities = {}
        self._relationships = {}

        # Load from disk if persist path is set
        if self._config.persist_path:
            self._load_from_disk()

        self._connected = True
        logger.info("In-memory graph backend initialized")

    def disconnect(self) -> None:
        """Save graph to disk if persist path is set."""
        if self._config.persist_path:
            self._save_to_disk()
        self._connected = False
        logger.info("In-memory graph backend disconnected")

    def is_connected(self) -> bool:
        """Check if the backend is initialized."""
        return self._connected

    def create_entity(self, entity: Entity) -> Entity:
        """Create a new entity in the graph."""
        entity_id = str(entity.id)
        self._entities[entity_id] = entity
        self._graph.add_node(entity_id, entity=entity)
        logger.debug(f"Created entity: {entity_id}")
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self._entities.get(entity_id)

    def update_entity(self, entity: Entity) -> Entity:
        """Update an existing entity."""
        entity_id = str(entity.id)
        if entity_id not in self._entities:
            raise EntityNotFoundError(entity_id)

        entity.updated_at = datetime.utcnow()
        self._entities[entity_id] = entity
        self._graph.nodes[entity_id]["entity"] = entity
        return entity

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID."""
        if entity_id not in self._entities:
            return False

        # Remove associated relationships
        to_remove = []
        for rel_id, rel in self._relationships.items():
            if str(rel.source_id) == entity_id or str(rel.target_id) == entity_id:
                to_remove.append(rel_id)

        for rel_id in to_remove:
            del self._relationships[rel_id]

        # Remove from graph
        self._graph.remove_node(entity_id)
        del self._entities[entity_id]
        return True

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Find entities matching criteria."""
        results = []
        for entity in self._entities.values():
            # Check entity type
            if entity_type:
                etype = (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else entity.entity_type
                )
                if etype != entity_type:
                    continue

            # Check properties
            if properties:
                match = True
                for key, value in properties.items():
                    if entity.properties.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(entity)
            if len(results) >= limit:
                break

        return results

    def create_relationship(
        self, source_id: str, target_id: str, relationship: Relationship
    ) -> Relationship:
        """Create a relationship between two entities."""
        if source_id not in self._entities:
            raise EntityNotFoundError(source_id)
        if target_id not in self._entities:
            raise EntityNotFoundError(target_id)

        rel_id = str(relationship.id)
        self._relationships[rel_id] = relationship

        # Add edge to graph
        rel_type = (
            relationship.relationship_type.value
            if hasattr(relationship.relationship_type, "value")
            else relationship.relationship_type
        )
        self._graph.add_edge(
            source_id,
            target_id,
            relationship=relationship,
            rel_type=rel_type,
            rel_id=rel_id,
        )

        # Add reverse edge if bidirectional
        if relationship.bidirectional:
            self._graph.add_edge(
                target_id,
                source_id,
                relationship=relationship,
                rel_type=rel_type,
                rel_id=rel_id,
            )

        logger.debug(f"Created relationship: {rel_id}")
        return relationship

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve a relationship by ID."""
        return self._relationships.get(relationship_id)

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        results = []

        if direction in ("outgoing", "both"):
            for _, target, data in self._graph.out_edges(entity_id, data=True):
                rel = data.get("relationship")
                if rel:
                    if relationship_type and data.get("rel_type") != relationship_type:
                        continue
                    if rel not in results:
                        results.append(rel)

        if direction in ("incoming", "both"):
            for source, _, data in self._graph.in_edges(entity_id, data=True):
                rel = data.get("relationship")
                if rel:
                    if relationship_type and data.get("rel_type") != relationship_type:
                        continue
                    if rel not in results:
                        results.append(rel)

        return results

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship by ID."""
        if relationship_id not in self._relationships:
            return False

        rel = self._relationships[relationship_id]
        source_id = str(rel.source_id)
        target_id = str(rel.target_id)

        # Remove edge(s)
        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)
        if rel.bidirectional and self._graph.has_edge(target_id, source_id):
            self._graph.remove_edge(target_id, source_id)

        del self._relationships[relationship_id]
        return True

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """
        Execute a query (limited support for in-memory backend).

        This method provides basic query support for testing.
        """
        # For in-memory backend, we support simple operations
        logger.warning("execute_query has limited support in memory backend")
        return {"entities": len(self._entities), "relationships": len(self._relationships)}

    def traverse(
        self,
        start_id: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
    ) -> TraversalResult:
        """Traverse the graph from a starting node."""
        start_entity = self.get_entity(start_id)
        if not start_entity:
            raise EntityNotFoundError(start_id)

        visited: Set[str] = {start_id}
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        paths: List[List[UUID]] = []

        # BFS traversal
        current_level = {start_id}
        for d in range(depth):
            next_level: Set[str] = set()
            for node_id in current_level:
                # Get outgoing edges
                for _, neighbor, data in self._graph.out_edges(node_id, data=True):
                    if relationship_types:
                        if data.get("rel_type") not in relationship_types:
                            continue

                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        neighbor_entity = self._entities.get(neighbor)
                        if neighbor_entity:
                            entities.append(neighbor_entity)

                    rel = data.get("relationship")
                    if rel and rel not in relationships:
                        relationships.append(rel)

                # Get incoming edges
                for neighbor, _, data in self._graph.in_edges(node_id, data=True):
                    if relationship_types:
                        if data.get("rel_type") not in relationship_types:
                            continue

                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        neighbor_entity = self._entities.get(neighbor)
                        if neighbor_entity:
                            entities.append(neighbor_entity)

                    rel = data.get("relationship")
                    if rel and rel not in relationships:
                        relationships.append(rel)

            current_level = next_level
            if not current_level:
                break

        return TraversalResult(
            start_entity=start_entity,
            entities=entities,
            relationships=relationships,
            depth_reached=depth,
        )

    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """Get neighboring entities."""
        result = self.traverse(entity_id, depth, relationship_types)
        return result.entities

    def _save_to_disk(self) -> None:
        """Save the graph to disk."""
        if not self._config.persist_path:
            return

        path = Path(self._config.persist_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save entities
        entities_data = {
            eid: entity.model_dump(mode="json") for eid, entity in self._entities.items()
        }
        with open(path / "entities.json", "w") as f:
            json.dump(entities_data, f, default=str)

        # Save relationships
        rels_data = {
            rid: rel.model_dump(mode="json") for rid, rel in self._relationships.items()
        }
        with open(path / "relationships.json", "w") as f:
            json.dump(rels_data, f, default=str)

        logger.info(f"Saved graph to {path}")

    def _load_from_disk(self) -> None:
        """Load the graph from disk."""
        if not self._config.persist_path:
            return

        path = Path(self._config.persist_path)
        if not path.exists():
            return

        # Load entities
        entities_file = path / "entities.json"
        if entities_file.exists():
            with open(entities_file) as f:
                entities_data = json.load(f)
            for eid, data in entities_data.items():
                entity = Entity(**data)
                self._entities[eid] = entity
                self._graph.add_node(eid, entity=entity)

        # Load relationships
        rels_file = path / "relationships.json"
        if rels_file.exists():
            with open(rels_file) as f:
                rels_data = json.load(f)
            for rid, data in rels_data.items():
                rel = Relationship(**data)
                self._relationships[rid] = rel
                source_id = str(rel.source_id)
                target_id = str(rel.target_id)
                rel_type = (
                    rel.relationship_type.value
                    if hasattr(rel.relationship_type, "value")
                    else rel.relationship_type
                )
                self._graph.add_edge(
                    source_id,
                    target_id,
                    relationship=rel,
                    rel_type=rel_type,
                    rel_id=rid,
                )

        logger.info(f"Loaded graph from {path}")
