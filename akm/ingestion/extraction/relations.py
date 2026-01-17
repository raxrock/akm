"""Relationship extraction from entities and content."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from akm.core.models import Entity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class RelationshipExtractionResult:
    """Result from relationship extraction."""

    relationships: List[Relationship] = field(default_factory=list)
    co_occurrences: List[Tuple[str, str, float]] = field(default_factory=list)


class RelationshipExtractor:
    """
    Extract relationships between entities.

    Supports:
    - Code-based relationships (imports, inheritance, calls)
    - Co-occurrence based relationships
    - LLM-based relationship extraction
    """

    # Code relationship patterns
    CODE_RELATIONSHIP_PATTERNS = {
        "python": {
            "inherits": r"class\s+(\w+)\s*\(\s*(\w+(?:\s*,\s*\w+)*)\s*\):",
            "imports_from": r"from\s+(\S+)\s+import\s+(\w+)",
            "imports": r"^import\s+(\S+)",
            "calls": r"(\w+)\s*\.\s*(\w+)\s*\(",
        },
        "javascript": {
            "imports_from": r"import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]",
            "extends": r"class\s+(\w+)\s+extends\s+(\w+)",
            "calls": r"(\w+)\s*\.\s*(\w+)\s*\(",
        },
        "java": {
            "extends": r"class\s+(\w+)\s+extends\s+(\w+)",
            "implements": r"class\s+(\w+)\s+implements\s+(\w+(?:\s*,\s*\w+)*)",
            "imports": r"import\s+(\S+);",
        },
        "go": {
            "imports": r"import\s+[\"']([^\"']+)[\"']",
            "calls": r"(\w+)\s*\.\s*(\w+)\s*\(",
        },
    }

    # Relationship type hierarchy
    RELATIONSHIP_WEIGHTS = {
        "inherits": 0.9,
        "extends": 0.9,
        "implements": 0.85,
        "imports": 0.7,
        "imports_from": 0.7,
        "calls": 0.6,
        "uses": 0.5,
        "references": 0.4,
        "co_occurs": 0.3,
    }

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: Optional[Any] = None,
        co_occurrence_window: int = 50,
        min_co_occurrence_score: float = 0.3,
    ) -> None:
        """
        Initialize the relationship extractor.

        Args:
            use_llm: Whether to use LLM for extraction
            llm_provider: LLM provider instance
            co_occurrence_window: Window size for co-occurrence detection
            min_co_occurrence_score: Minimum score for co-occurrence relationships
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.co_occurrence_window = co_occurrence_window
        self.min_co_occurrence_score = min_co_occurrence_score

    def extract(
        self,
        content: str,
        entities: List[Entity],
        language: Optional[str] = None,
        source_path: str = "",
    ) -> RelationshipExtractionResult:
        """
        Extract relationships from content and entities.

        Args:
            content: Source content
            entities: Previously extracted entities
            language: Programming language (if applicable)
            source_path: Source path for context

        Returns:
            RelationshipExtractionResult with found relationships
        """
        result = RelationshipExtractionResult()

        # Extract code-based relationships if language specified
        if language:
            code_rels = self._extract_code_relationships(
                content, entities, language, source_path
            )
            result.relationships.extend(code_rels)

        # Extract co-occurrence relationships
        co_rels, co_occs = self._extract_co_occurrences(content, entities)
        result.relationships.extend(co_rels)
        result.co_occurrences = co_occs

        # Use LLM for additional extraction if configured
        if self.use_llm and self.llm_provider:
            llm_rels = self._extract_with_llm(content, entities)
            result.relationships.extend(llm_rels)

        # Deduplicate relationships
        result.relationships = self._deduplicate_relationships(result.relationships)

        return result

    def _extract_code_relationships(
        self,
        content: str,
        entities: List[Entity],
        language: str,
        source_path: str,
    ) -> List[Relationship]:
        """Extract relationships from code patterns."""
        relationships = []
        patterns = self.CODE_RELATIONSHIP_PATTERNS.get(language, {})

        # Build entity name lookup
        entity_map = {e.name: e for e in entities}

        for rel_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE)

            for match in matches:
                groups = [g for g in match.groups() if g]
                if len(groups) >= 2:
                    source_name = groups[0].strip()
                    target_names = groups[1] if len(groups) > 1 else ""

                    # Handle multiple targets (e.g., multiple inheritance)
                    for target_name in target_names.split(","):
                        target_name = target_name.strip()
                        if not target_name:
                            continue

                        # Try to match with known entities
                        source_entity = entity_map.get(source_name)
                        target_entity = entity_map.get(target_name)

                        if source_entity and target_entity:
                            rel = Relationship(
                                source_id=str(source_entity.id),
                                target_id=str(target_entity.id),
                                relationship_type=rel_type.upper(),
                                properties={
                                    "source_path": source_path,
                                    "pattern_matched": pattern,
                                    "weight": self.RELATIONSHIP_WEIGHTS.get(rel_type, 0.5),
                                },
                            )
                            relationships.append(rel)
                elif len(groups) == 1:
                    # Single target relationship (e.g., simple import)
                    target_name = groups[0].strip()
                    # Find the source (usually the file/module itself)
                    source_entity = self._find_file_entity(entities, source_path)
                    # Try to find target entity - only create relationship if it exists
                    target_entity = entity_map.get(target_name)

                    if source_entity and target_entity:
                        rel = Relationship(
                            source_id=str(source_entity.id),
                            target_id=str(target_entity.id),
                            relationship_type=rel_type.upper(),
                            properties={
                                "source_path": source_path,
                                "weight": self.RELATIONSHIP_WEIGHTS.get(rel_type, 0.5),
                            },
                        )
                        relationships.append(rel)
                    elif source_entity and target_name:
                        # Create entity for external dependency
                        from uuid import uuid4
                        external_entity = Entity(
                            id=uuid4(),
                            name=target_name,
                            entity_type="external_dependency",
                            properties={
                                "source_path": source_path,
                                "external": True,
                            },
                        )
                        # Add to entity map for future reference
                        entity_map[target_name] = external_entity
                        entities.append(external_entity)

                        rel = Relationship(
                            source_id=str(source_entity.id),
                            target_id=str(external_entity.id),
                            relationship_type=rel_type.upper(),
                            properties={
                                "source_path": source_path,
                                "weight": self.RELATIONSHIP_WEIGHTS.get(rel_type, 0.5),
                                "target_external": True,
                            },
                        )
                        relationships.append(rel)

        return relationships

    def _extract_co_occurrences(
        self,
        content: str,
        entities: List[Entity],
    ) -> Tuple[List[Relationship], List[Tuple[str, str, float]]]:
        """
        Extract co-occurrence relationships.

        Two entities are considered co-occurring if they appear
        within a certain window of each other.
        """
        relationships = []
        co_occurrences = []

        # Find positions of all entity mentions
        entity_positions: Dict[str, List[int]] = {}
        for entity in entities:
            name = entity.name
            positions = [m.start() for m in re.finditer(re.escape(name), content)]
            if positions:
                entity_positions[str(entity.id)] = positions

        # Calculate co-occurrence scores
        entity_ids = list(entity_positions.keys())
        entity_id_to_entity = {str(e.id): e for e in entities}

        for i, id1 in enumerate(entity_ids):
            for id2 in entity_ids[i + 1:]:
                score = self._calculate_co_occurrence_score(
                    entity_positions[id1],
                    entity_positions[id2],
                    len(content),
                )

                if score >= self.min_co_occurrence_score:
                    name1 = entity_id_to_entity.get(id1, Entity(name="unknown")).name
                    name2 = entity_id_to_entity.get(id2, Entity(name="unknown")).name
                    co_occurrences.append((name1, name2, score))

                    rel = Relationship(
                        source_id=id1,
                        target_id=id2,
                        relationship_type="CO_OCCURS",
                        properties={
                            "score": score,
                            "pattern_source": "co_occurrence",
                            "weight": score * 0.5,  # Scale down for soft links
                        },
                    )
                    relationships.append(rel)

        return relationships, co_occurrences

    def _calculate_co_occurrence_score(
        self,
        positions1: List[int],
        positions2: List[int],
        content_length: int,
    ) -> float:
        """
        Calculate co-occurrence score between two entities.

        Score is based on:
        - Number of co-occurrences within window
        - Proximity of occurrences
        - Frequency normalization
        """
        if not positions1 or not positions2:
            return 0.0

        co_occurrence_count = 0
        total_proximity = 0.0

        for pos1 in positions1:
            for pos2 in positions2:
                distance = abs(pos1 - pos2)
                if distance <= self.co_occurrence_window:
                    co_occurrence_count += 1
                    # Closer = higher score
                    proximity = 1.0 - (distance / self.co_occurrence_window)
                    total_proximity += proximity

        if co_occurrence_count == 0:
            return 0.0

        # Normalize by total possible co-occurrences
        max_possible = len(positions1) * len(positions2)
        frequency_score = co_occurrence_count / max_possible

        # Average proximity
        avg_proximity = total_proximity / co_occurrence_count

        # Combined score
        return (frequency_score + avg_proximity) / 2

    def _extract_with_llm(
        self,
        content: str,
        entities: List[Entity],
    ) -> List[Relationship]:
        """Use LLM to extract relationships."""
        if not self.llm_provider:
            return []

        try:
            entity_names = [e.name for e in entities[:20]]  # Limit for context
            # Build entity lookup by name
            entity_by_name = {e.name.lower(): e for e in entities}

            extracted = self.llm_provider.extract_relationships(content, entity_names)

            # Convert ExtractedRelationship to Relationship
            relationships = []
            for ext_rel in extracted:
                # Get entity text from ExtractedRelationship
                source_text = getattr(ext_rel.source_entity, 'text', str(ext_rel.source_entity))
                target_text = getattr(ext_rel.target_entity, 'text', str(ext_rel.target_entity))

                # Find matching entities
                source_entity = entity_by_name.get(source_text.lower())
                target_entity = entity_by_name.get(target_text.lower())

                if source_entity and target_entity:
                    relationships.append(
                        Relationship(
                            source_id=str(source_entity.id),
                            target_id=str(target_entity.id),
                            relationship_type=ext_rel.relationship_type,
                            properties={
                                "confidence": ext_rel.confidence,
                                "source": "llm",
                            },
                        )
                    )

            return relationships
        except Exception as e:
            logger.warning(f"LLM relationship extraction failed: {e}")
            return []

    def _find_file_entity(
        self,
        entities: List[Entity],
        source_path: str,
    ) -> Optional[Entity]:
        """Find the entity representing a file."""
        for entity in entities:
            if entity.entity_type in ("file", "module", "CodeFile"):
                if entity.properties.get("source_path") == source_path:
                    return entity
        return None

    def _deduplicate_relationships(
        self,
        relationships: List[Relationship],
    ) -> List[Relationship]:
        """Remove duplicate relationships, keeping highest weight."""
        seen: Dict[Tuple[str, str, str], Relationship] = {}

        for rel in relationships:
            key = (rel.source_id, rel.target_id, rel.relationship_type)
            if key not in seen:
                seen[key] = rel
            else:
                # Keep the one with higher weight
                existing_weight = seen[key].properties.get("weight", 0)
                new_weight = rel.properties.get("weight", 0)
                if new_weight > existing_weight:
                    seen[key] = rel

        return list(seen.values())


__all__ = ["RelationshipExtractor", "RelationshipExtractionResult"]
