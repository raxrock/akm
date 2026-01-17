"""Pattern-based soft link creation for adaptive knowledge mesh."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

from akm.core.models import Entity, Link, LinkStatus, LinkWeight

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a detected pattern that may indicate a link."""

    source_entity_id: str
    target_entity_id: str
    pattern_type: str
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None


@dataclass
class CoOccurrencePattern:
    """Pattern based on entity co-occurrence in documents."""

    entity_ids: Set[str]
    document_ids: Set[str]
    occurrence_count: int
    window_size: int = 100  # Characters within which entities co-occur

    @property
    def strength(self) -> float:
        """Calculate pattern strength based on co-occurrence frequency."""
        # More occurrences and appearances in multiple documents = stronger
        doc_factor = min(1.0, len(self.document_ids) / 5)
        count_factor = min(1.0, self.occurrence_count / 10)
        return (doc_factor + count_factor) / 2


@dataclass
class SemanticPattern:
    """Pattern based on semantic similarity between entities."""

    source_entity_id: str
    target_entity_id: str
    similarity_score: float
    embedding_model: str = "unknown"

    @property
    def strength(self) -> float:
        """Pattern strength is the similarity score."""
        return self.similarity_score


@dataclass
class StructuralPattern:
    """Pattern based on structural relationships in documents."""

    source_entity_id: str
    target_entity_id: str
    structure_type: str  # "same_section", "parent_child", "sibling", etc.
    document_id: str
    confidence: float = 0.5


class PatternDetector:
    """
    Detects patterns that may indicate relationships between entities.

    Pattern detection strategies:
    1. Co-occurrence: Entities appearing near each other in text
    2. Semantic similarity: Entities with similar embeddings
    3. Structural: Entities in related document structures
    4. Explicit mentions: Text explicitly mentioning relationships
    """

    def __init__(
        self,
        co_occurrence_window: int = 200,
        min_co_occurrence_count: int = 2,
        semantic_similarity_threshold: float = 0.6,
    ) -> None:
        """
        Initialize the pattern detector.

        Args:
            co_occurrence_window: Character window for co-occurrence detection
            min_co_occurrence_count: Minimum times entities must co-occur
            semantic_similarity_threshold: Minimum similarity for semantic patterns
        """
        self.co_occurrence_window = co_occurrence_window
        self.min_co_occurrence_count = min_co_occurrence_count
        self.semantic_similarity_threshold = semantic_similarity_threshold

        # Track co-occurrences
        self._co_occurrences: Dict[Tuple[str, str], CoOccurrencePattern] = {}

        # Relationship indicator phrases
        self._relationship_phrases = [
            (r"(\w+)\s+(?:is|was|were)\s+(?:created|developed|built|designed)\s+by\s+(\w+)", "CREATED_BY"),
            (r"(\w+)\s+(?:uses|utilizes|leverages|relies on)\s+(\w+)", "USES"),
            (r"(\w+)\s+(?:depends on|requires)\s+(\w+)", "DEPENDS_ON"),
            (r"(\w+)\s+(?:is part of|belongs to)\s+(\w+)", "PART_OF"),
            (r"(\w+)\s+(?:leads?|manages?|owns?)\s+(\w+)", "MANAGES"),
            (r"(\w+)\s+(?:works? (?:on|with)|collaborates? (?:on|with))\s+(\w+)", "WORKS_ON"),
            (r"(\w+)\s+(?:is responsible for|handles?)\s+(\w+)", "RESPONSIBLE_FOR"),
            (r"(\w+)\s+(?:integrates? with|connects? to)\s+(\w+)", "INTEGRATES_WITH"),
        ]

    def detect_co_occurrence(
        self,
        entities: List[Entity],
        text: str,
        document_id: str,
    ) -> List[CoOccurrencePattern]:
        """
        Detect co-occurrence patterns in text.

        Args:
            entities: List of entities to check
            text: Document text
            document_id: Document identifier

        Returns:
            List of detected co-occurrence patterns
        """
        patterns = []

        # Find entity positions in text
        entity_positions: Dict[str, List[int]] = {}
        for entity in entities:
            # Simple case-insensitive search for entity name
            name_lower = entity.name.lower()
            text_lower = text.lower()
            pos = 0
            while True:
                idx = text_lower.find(name_lower, pos)
                if idx == -1:
                    break
                if str(entity.id) not in entity_positions:
                    entity_positions[str(entity.id)] = []
                entity_positions[str(entity.id)].append(idx)
                pos = idx + 1

        # Find co-occurring pairs
        entity_ids = list(entity_positions.keys())
        for i, eid1 in enumerate(entity_ids):
            for eid2 in entity_ids[i + 1:]:
                positions1 = entity_positions[eid1]
                positions2 = entity_positions[eid2]

                # Count co-occurrences within window
                co_occur_count = 0
                for p1 in positions1:
                    for p2 in positions2:
                        if abs(p1 - p2) <= self.co_occurrence_window:
                            co_occur_count += 1

                if co_occur_count > 0:
                    # Create or update pattern
                    key = (min(eid1, eid2), max(eid1, eid2))
                    if key not in self._co_occurrences:
                        self._co_occurrences[key] = CoOccurrencePattern(
                            entity_ids={eid1, eid2},
                            document_ids={document_id},
                            occurrence_count=co_occur_count,
                            window_size=self.co_occurrence_window,
                        )
                    else:
                        self._co_occurrences[key].document_ids.add(document_id)
                        self._co_occurrences[key].occurrence_count += co_occur_count

                    if self._co_occurrences[key].occurrence_count >= self.min_co_occurrence_count:
                        patterns.append(self._co_occurrences[key])

        return patterns

    def detect_semantic_patterns(
        self,
        entities: List[Entity],
        compute_similarity: Callable[[List[float], List[float]], float],
    ) -> List[SemanticPattern]:
        """
        Detect semantic similarity patterns between entities.

        Args:
            entities: List of entities with embeddings
            compute_similarity: Function to compute similarity between embeddings

        Returns:
            List of detected semantic patterns
        """
        patterns = []
        entities_with_embeddings = [e for e in entities if e.embedding]

        for i, e1 in enumerate(entities_with_embeddings):
            for e2 in entities_with_embeddings[i + 1:]:
                if e1.embedding and e2.embedding:
                    similarity = compute_similarity(e1.embedding, e2.embedding)

                    if similarity >= self.semantic_similarity_threshold:
                        patterns.append(
                            SemanticPattern(
                                source_entity_id=str(e1.id),
                                target_entity_id=str(e2.id),
                                similarity_score=similarity,
                            )
                        )

        return patterns

    def detect_explicit_relationships(
        self,
        entities: List[Entity],
        text: str,
    ) -> List[PatternMatch]:
        """
        Detect explicitly mentioned relationships in text.

        Args:
            entities: List of entities
            text: Document text

        Returns:
            List of pattern matches for explicit relationships
        """
        patterns = []
        entity_name_map = {e.name.lower(): str(e.id) for e in entities}

        for pattern_regex, rel_type in self._relationship_phrases:
            for match in re.finditer(pattern_regex, text, re.IGNORECASE):
                source_text = match.group(1).lower()
                target_text = match.group(2).lower()

                # Try to match to entities
                source_id = entity_name_map.get(source_text)
                target_id = entity_name_map.get(target_text)

                if source_id and target_id and source_id != target_id:
                    patterns.append(
                        PatternMatch(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            pattern_type=f"explicit_{rel_type.lower()}",
                            confidence=0.8,
                            evidence={
                                "matched_text": match.group(0),
                                "relationship_type": rel_type,
                            },
                            context=text[max(0, match.start() - 50):match.end() + 50],
                        )
                    )

        return patterns

    def detect_structural_patterns(
        self,
        entities: List[Entity],
        sections: List[Dict[str, Any]],
        document_id: str,
    ) -> List[StructuralPattern]:
        """
        Detect patterns based on document structure.

        Args:
            entities: List of entities
            sections: Document sections with entities
            document_id: Document identifier

        Returns:
            List of structural patterns
        """
        patterns = []

        # Group entities by section
        section_entities: Dict[str, List[str]] = defaultdict(list)
        for section in sections:
            section_id = section.get("id", "")
            for entity in section.get("entities", []):
                section_entities[section_id].append(str(entity.id))

        # Entities in same section are potentially related
        for section_id, entity_ids in section_entities.items():
            for i, eid1 in enumerate(entity_ids):
                for eid2 in entity_ids[i + 1:]:
                    patterns.append(
                        StructuralPattern(
                            source_entity_id=eid1,
                            target_entity_id=eid2,
                            structure_type="same_section",
                            document_id=document_id,
                            confidence=0.4,
                        )
                    )

        return patterns

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected patterns."""
        return {
            "co_occurrence_pairs": len(self._co_occurrences),
            "total_co_occurrences": sum(
                p.occurrence_count for p in self._co_occurrences.values()
            ),
        }


class SoftLinkCreator:
    """
    Creates soft links from detected patterns.

    Soft links are unvalidated connections that may become
    validated relationships through user interaction or other signals.
    """

    def __init__(
        self,
        initial_weight: float = 0.3,
        decay_rate: float = 0.01,
        promotion_threshold: float = 0.8,
        demotion_threshold: float = 0.1,
    ) -> None:
        """
        Initialize the soft link creator.

        Args:
            initial_weight: Initial weight for new soft links
            decay_rate: Decay rate for link weights
            promotion_threshold: Weight threshold for promotion
            demotion_threshold: Weight threshold for demotion
        """
        self.initial_weight = initial_weight
        self.decay_rate = decay_rate
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold

    def create_from_co_occurrence(
        self,
        pattern: CoOccurrencePattern,
    ) -> Link:
        """
        Create a soft link from a co-occurrence pattern.

        Args:
            pattern: Co-occurrence pattern

        Returns:
            New soft link
        """
        entity_ids = list(pattern.entity_ids)
        source_id = entity_ids[0]
        target_id = entity_ids[1]

        # Weight based on pattern strength
        weight_value = min(0.6, self.initial_weight + pattern.strength * 0.3)

        weight = LinkWeight(
            value=weight_value,
            initial_value=weight_value,
            decay_rate=self.decay_rate,
            promotion_threshold=self.promotion_threshold,
            demotion_threshold=self.demotion_threshold,
        )

        return Link(
            source_id=UUID(source_id),
            target_id=UUID(target_id),
            link_type="co_occurrence",
            status=LinkStatus.SOFT,
            weight=weight,
            pattern_source="co_occurrence_detector",
            pattern_confidence=pattern.strength,
            co_occurrence_count=pattern.occurrence_count,
            metadata={
                "document_ids": list(pattern.document_ids),
                "window_size": pattern.window_size,
            },
        )

    def create_from_semantic_similarity(
        self,
        pattern: SemanticPattern,
    ) -> Link:
        """
        Create a soft link from a semantic similarity pattern.

        Args:
            pattern: Semantic similarity pattern

        Returns:
            New soft link
        """
        # Higher similarity = higher initial weight
        weight_value = min(0.7, self.initial_weight + pattern.similarity_score * 0.4)

        weight = LinkWeight(
            value=weight_value,
            initial_value=weight_value,
            decay_rate=self.decay_rate,
            promotion_threshold=self.promotion_threshold,
            demotion_threshold=self.demotion_threshold,
        )

        return Link(
            source_id=UUID(pattern.source_entity_id),
            target_id=UUID(pattern.target_entity_id),
            link_type="semantic_similarity",
            status=LinkStatus.SOFT,
            weight=weight,
            pattern_source="semantic_detector",
            pattern_confidence=pattern.similarity_score,
            semantic_similarity=pattern.similarity_score,
            metadata={
                "embedding_model": pattern.embedding_model,
            },
        )

    def create_from_explicit_mention(
        self,
        pattern: PatternMatch,
    ) -> Link:
        """
        Create a soft link from an explicit relationship mention.

        Args:
            pattern: Pattern match from explicit mention

        Returns:
            New soft link
        """
        # Explicit mentions get higher initial weight
        weight_value = min(0.8, self.initial_weight + pattern.confidence * 0.5)

        weight = LinkWeight(
            value=weight_value,
            initial_value=weight_value,
            decay_rate=self.decay_rate * 0.5,  # Slower decay for explicit mentions
            promotion_threshold=self.promotion_threshold,
            demotion_threshold=self.demotion_threshold,
        )

        return Link(
            source_id=UUID(pattern.source_entity_id),
            target_id=UUID(pattern.target_entity_id),
            link_type=pattern.pattern_type,
            status=LinkStatus.SOFT,
            weight=weight,
            pattern_source="explicit_mention_detector",
            pattern_confidence=pattern.confidence,
            metadata={
                "evidence": pattern.evidence,
                "context": pattern.context,
            },
        )

    def create_from_structural_pattern(
        self,
        pattern: StructuralPattern,
    ) -> Link:
        """
        Create a soft link from a structural pattern.

        Args:
            pattern: Structural pattern

        Returns:
            New soft link
        """
        weight_value = self.initial_weight + pattern.confidence * 0.2

        weight = LinkWeight(
            value=weight_value,
            initial_value=weight_value,
            decay_rate=self.decay_rate,
            promotion_threshold=self.promotion_threshold,
            demotion_threshold=self.demotion_threshold,
        )

        return Link(
            source_id=UUID(pattern.source_entity_id),
            target_id=UUID(pattern.target_entity_id),
            link_type=f"structural_{pattern.structure_type}",
            status=LinkStatus.SOFT,
            weight=weight,
            pattern_source="structural_detector",
            pattern_confidence=pattern.confidence,
            metadata={
                "structure_type": pattern.structure_type,
                "document_id": pattern.document_id,
            },
        )

    def merge_links(self, existing: Link, new: Link) -> Link:
        """
        Merge a new link detection into an existing link.

        Args:
            existing: Existing link
            new: Newly detected link

        Returns:
            Updated link
        """
        # Increment co-occurrence count
        existing.co_occurrence_count += new.co_occurrence_count

        # Update weight with weighted average
        existing.weight.value = min(
            1.0,
            existing.weight.value * 0.7 + new.weight.value * 0.3
        )

        # Update semantic similarity if provided
        if new.semantic_similarity:
            if existing.semantic_similarity:
                existing.semantic_similarity = (
                    existing.semantic_similarity * 0.7 + new.semantic_similarity * 0.3
                )
            else:
                existing.semantic_similarity = new.semantic_similarity

        # Merge metadata
        existing.metadata.update(new.metadata)

        return existing


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0-1)
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def detect_and_create_soft_links(
    entities: List[Entity],
    text: str,
    document_id: str,
    pattern_detector: Optional[PatternDetector] = None,
    link_creator: Optional[SoftLinkCreator] = None,
) -> List[Link]:
    """
    Detect patterns and create soft links from a document.

    Args:
        entities: Entities found in the document
        text: Document text
        document_id: Document identifier
        pattern_detector: Optional custom pattern detector
        link_creator: Optional custom link creator

    Returns:
        List of created soft links
    """
    detector = pattern_detector or PatternDetector()
    creator = link_creator or SoftLinkCreator()

    links = []

    # Detect co-occurrence patterns
    co_occurrence_patterns = detector.detect_co_occurrence(entities, text, document_id)
    for pattern in co_occurrence_patterns:
        links.append(creator.create_from_co_occurrence(pattern))

    # Detect semantic patterns
    semantic_patterns = detector.detect_semantic_patterns(
        entities,
        compute_cosine_similarity,
    )
    for pattern in semantic_patterns:
        links.append(creator.create_from_semantic_similarity(pattern))

    # Detect explicit relationship mentions
    explicit_patterns = detector.detect_explicit_relationships(entities, text)
    for pattern in explicit_patterns:
        links.append(creator.create_from_explicit_mention(pattern))

    logger.info(
        f"Created {len(links)} soft links from document {document_id}: "
        f"{len(co_occurrence_patterns)} co-occurrence, "
        f"{len(semantic_patterns)} semantic, "
        f"{len(explicit_patterns)} explicit"
    )

    return links
