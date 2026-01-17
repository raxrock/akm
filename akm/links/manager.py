"""Link manager for orchestrating adaptive link lifecycle."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from akm.core.config import LinkConfig
from akm.core.exceptions import LinkNotFoundError
from akm.core.interfaces import GraphBackend
from akm.core.models import Entity, Link, LinkStatus, LinkWeight, Relationship
from akm.links.decay import decay_link, should_decay
from akm.links.validation import should_auto_validate, update_link_status, validate_link

logger = logging.getLogger(__name__)


class LinkManager:
    """
    Manages the adaptive link lifecycle.

    The LinkManager is responsible for:
    - Creating soft links from detected patterns
    - Validating links through user interactions
    - Applying time-based decay
    - Promoting/demoting links based on weight thresholds
    - Storing and retrieving links
    """

    def __init__(
        self,
        graph: GraphBackend,
        config: LinkConfig,
    ) -> None:
        """
        Initialize the link manager.

        Args:
            graph: Graph backend for storing links
            config: Link lifecycle configuration
        """
        self._graph = graph
        self._config = config
        self._links: Dict[str, Link] = {}  # In-memory cache of links
        self._entity_links: Dict[str, List[str]] = {}  # entity_id -> link_ids

    def create_soft_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "inferred",
        pattern_source: Optional[str] = None,
        pattern_confidence: float = 0.5,
        semantic_similarity: Optional[float] = None,
        **kwargs: Any,
    ) -> Link:
        """
        Create a new soft (unvalidated) link between entities.

        Soft links are created when the system detects a potential relationship
        through patterns, co-occurrence, or semantic similarity.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            link_type: Type of link (e.g., "co_occurrence", "semantic", "inferred")
            pattern_source: What detected this pattern
            pattern_confidence: Confidence in the pattern (0-1)
            semantic_similarity: Optional semantic similarity score
            **kwargs: Additional link properties

        Returns:
            The created Link
        """
        # Check if link already exists
        existing = self.get_link_between(source_id, target_id)
        if existing:
            # Increment co-occurrence and boost weight slightly
            existing.increment_co_occurrence()
            if semantic_similarity:
                # Update semantic similarity with weighted average
                if existing.semantic_similarity:
                    existing.semantic_similarity = (
                        existing.semantic_similarity * 0.7 + semantic_similarity * 0.3
                    )
                else:
                    existing.semantic_similarity = semantic_similarity
            logger.debug(f"Updated existing link: {existing.id}")
            return existing

        # Create new link
        weight = LinkWeight(
            value=self._config.initial_soft_link_weight,
            initial_value=self._config.initial_soft_link_weight,
            decay_rate=self._config.decay.decay_rate,
            promotion_threshold=self._config.promotion_threshold,
            demotion_threshold=self._config.demotion_threshold,
        )

        link = Link(
            id=uuid4(),
            source_id=UUID(source_id),
            target_id=UUID(target_id),
            link_type=link_type,
            status=LinkStatus.SOFT,
            weight=weight,
            pattern_source=pattern_source,
            pattern_confidence=pattern_confidence,
            semantic_similarity=semantic_similarity,
            co_occurrence_count=1,
            **kwargs,
        )

        # Check for auto-validation
        if should_auto_validate(link, self._config.validation):
            link.status = LinkStatus.VALIDATED
            link.weight.value = min(
                1.0, link.weight.value + self._config.validation.positive_weight_boost
            )

        # Store the link
        self._store_link(link)

        logger.info(
            f"Created soft link: {link.id} ({source_id} -> {target_id}) "
            f"with weight {link.weight.value:.3f}"
        )
        return link

    def validate_link(
        self,
        link_id: str,
        is_positive: bool,
        strength: Optional[float] = None,
    ) -> Link:
        """
        Validate a link through user interaction.

        Args:
            link_id: The link ID
            is_positive: Whether the validation is positive
            strength: Optional override for validation strength

        Returns:
            The updated Link
        """
        link = self.get_link(link_id)
        if not link:
            raise LinkNotFoundError(link_id)

        updated = validate_link(link, is_positive, self._config.validation, strength)
        self._store_link(updated)

        logger.info(
            f"Validated link: {link_id} (positive={is_positive}) "
            f"new weight: {updated.weight.value:.3f}"
        )
        return updated

    def get_link(self, link_id: str) -> Optional[Link]:
        """Get a link by ID."""
        return self._links.get(link_id)

    def get_link_between(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[Link]:
        """Get a link between two entities."""
        source_links = self._entity_links.get(source_id, [])
        for link_id in source_links:
            link = self._links.get(link_id)
            if link and str(link.target_id) == target_id:
                return link

        # Check reverse direction
        target_links = self._entity_links.get(target_id, [])
        for link_id in target_links:
            link = self._links.get(link_id)
            if link and str(link.source_id) == target_id and str(link.target_id) == source_id:
                return link

        return None

    def get_links(
        self,
        entity_id: str,
        min_weight: float = 0.0,
        status: Optional[LinkStatus] = None,
    ) -> List[Link]:
        """
        Get all links for an entity.

        Args:
            entity_id: The entity ID
            min_weight: Minimum weight threshold
            status: Optional status filter

        Returns:
            List of links
        """
        link_ids = self._entity_links.get(entity_id, [])
        results = []

        for link_id in link_ids:
            link = self._links.get(link_id)
            if link:
                if link.weight.value < min_weight:
                    continue
                if status and link.status != status:
                    continue
                results.append(link)

        return results

    def get_links_in_subgraph(
        self,
        entity_ids: List[str],
        min_weight: float = 0.0,
    ) -> List[Link]:
        """
        Get all links within a subgraph of entities.

        Args:
            entity_ids: List of entity IDs
            min_weight: Minimum weight threshold

        Returns:
            List of links within the subgraph
        """
        entity_set = set(entity_ids)
        results = []
        seen_ids = set()

        for entity_id in entity_ids:
            for link in self.get_links(entity_id, min_weight):
                if str(link.id) in seen_ids:
                    continue
                # Both ends must be in the subgraph
                if str(link.source_id) in entity_set and str(link.target_id) in entity_set:
                    results.append(link)
                    seen_ids.add(str(link.id))

        return results

    def run_decay(
        self,
        current_time: Optional[datetime] = None,
    ) -> int:
        """
        Run time-based decay on all links.

        Args:
            current_time: Current time (defaults to UTC now)

        Returns:
            Number of links that were decayed
        """
        if not self._config.decay.enabled:
            return 0

        current_time = current_time or datetime.now(timezone.utc)
        decayed_count = 0

        for link_id, link in list(self._links.items()):
            if should_decay(link, self._config.decay, current_time):
                decay_link(link, self._config.decay, current_time)
                decayed_count += 1

                # Remove if archived
                if link.status == LinkStatus.ARCHIVED:
                    logger.info(f"Archived link {link_id} (weight: {link.weight.value:.3f})")

        logger.info(f"Decay run complete: {decayed_count} links processed")
        return decayed_count

    def get_strong_links(
        self,
        min_weight: float = 0.7,
        status: Optional[LinkStatus] = None,
    ) -> List[Link]:
        """Get all links above a weight threshold."""
        results = []
        for link in self._links.values():
            if link.weight.value >= min_weight:
                if status is None or link.status == status:
                    results.append(link)
        return results

    def get_weak_links(
        self,
        max_weight: float = 0.3,
    ) -> List[Link]:
        """Get all links below a weight threshold."""
        return [
            link
            for link in self._links.values()
            if link.weight.value <= max_weight and link.status != LinkStatus.ARCHIVED
        ]

    def delete_link(self, link_id: str) -> bool:
        """Delete a link by ID."""
        link = self._links.get(link_id)
        if not link:
            return False

        # Remove from entity index
        source_id = str(link.source_id)
        target_id = str(link.target_id)

        if source_id in self._entity_links:
            self._entity_links[source_id] = [
                lid for lid in self._entity_links[source_id] if lid != link_id
            ]
        if target_id in self._entity_links:
            self._entity_links[target_id] = [
                lid for lid in self._entity_links[target_id] if lid != link_id
            ]

        del self._links[link_id]
        return True

    def cleanup_archived_links(self) -> int:
        """Remove all archived links."""
        to_remove = [
            link_id
            for link_id, link in self._links.items()
            if link.status == LinkStatus.ARCHIVED
        ]

        for link_id in to_remove:
            self.delete_link(link_id)

        logger.info(f"Cleaned up {len(to_remove)} archived links")
        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the link manager."""
        status_counts = {}
        weight_sum = 0.0
        weight_count = 0

        for link in self._links.values():
            status = link.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            weight_sum += link.weight.value
            weight_count += 1

        return {
            "total_links": len(self._links),
            "status_distribution": status_counts,
            "average_weight": weight_sum / weight_count if weight_count > 0 else 0,
            "entities_with_links": len(self._entity_links),
        }

    def _store_link(self, link: Link) -> None:
        """Store a link in the internal cache."""
        link_id = str(link.id)
        source_id = str(link.source_id)
        target_id = str(link.target_id)

        self._links[link_id] = link

        # Update entity index
        if source_id not in self._entity_links:
            self._entity_links[source_id] = []
        if link_id not in self._entity_links[source_id]:
            self._entity_links[source_id].append(link_id)

        if target_id not in self._entity_links:
            self._entity_links[target_id] = []
        if link_id not in self._entity_links[target_id]:
            self._entity_links[target_id].append(link_id)

    def create_links_from_relationships(
        self,
        relationships: List[Relationship],
        pattern_source: str = "relationship",
    ) -> List[Link]:
        """
        Create soft links from existing relationships.

        This is useful for bootstrapping links from known relationships.

        Args:
            relationships: List of relationships to create links from
            pattern_source: Source identifier for these links

        Returns:
            List of created links
        """
        links = []
        for rel in relationships:
            link = self.create_soft_link(
                source_id=str(rel.source_id),
                target_id=str(rel.target_id),
                link_type=f"from_{rel.relationship_type}",
                pattern_source=pattern_source,
                pattern_confidence=rel.confidence,
            )
            links.append(link)
        return links

    def find_potential_links(
        self,
        entity: Entity,
        candidates: List[Entity],
        similarity_threshold: float = 0.5,
    ) -> List[Link]:
        """
        Find potential links between an entity and candidates.

        This uses simple heuristics; GNN-based prediction is handled separately.

        Args:
            entity: The entity to find links for
            candidates: Candidate entities
            similarity_threshold: Minimum similarity for link creation

        Returns:
            List of potential links (as soft links)
        """
        # This is a placeholder for basic heuristic-based link discovery
        # Real implementation would use embeddings/semantic similarity
        links = []

        for candidate in candidates:
            if str(candidate.id) == str(entity.id):
                continue

            # Check if entities share properties
            shared_props = set(entity.properties.keys()) & set(candidate.properties.keys())
            if shared_props:
                confidence = len(shared_props) / max(
                    len(entity.properties), len(candidate.properties), 1
                )
                if confidence >= similarity_threshold:
                    link = self.create_soft_link(
                        source_id=str(entity.id),
                        target_id=str(candidate.id),
                        link_type="property_similarity",
                        pattern_source="heuristic",
                        pattern_confidence=confidence,
                    )
                    links.append(link)

        return links
