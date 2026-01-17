"""Named Entity Recognition for entity extraction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from akm.core.models import Entity

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from entity extraction."""

    entities: List[Entity] = field(default_factory=list)
    raw_mentions: List[Dict[str, Any]] = field(default_factory=list)
    source_path: str = ""
    content_type: str = ""


class EntityExtractor:
    """
    Extract entities from text using various methods.

    Supports:
    - Pattern-based extraction (regex)
    - Code structure extraction (for code files)
    - LLM-based extraction (if LLM provider configured)
    """

    # Code patterns for different languages
    CODE_PATTERNS = {
        "python": {
            "class": r"class\s+(\w+)\s*(?:\([^)]*\))?:",
            "function": r"def\s+(\w+)\s*\([^)]*\):",
            "import": r"(?:from\s+(\S+)\s+)?import\s+([^\n;]+)",
            "variable": r"^(\w+)\s*=\s*(?!.*def\s|.*class\s)",
        },
        "javascript": {
            "class": r"class\s+(\w+)\s*(?:extends\s+\w+\s*)?\{",
            "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
            "import": r"import\s+(?:\{[^}]+\}|[\w*]+)\s+from\s+['\"]([^'\"]+)['\"]",
        },
        "java": {
            "class": r"(?:public|private|protected)?\s*class\s+(\w+)",
            "interface": r"(?:public|private|protected)?\s*interface\s+(\w+)",
            "method": r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\([^)]*\)",
        },
        "go": {
            "struct": r"type\s+(\w+)\s+struct\s*\{",
            "interface": r"type\s+(\w+)\s+interface\s*\{",
            "function": r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)",
        },
    }

    # Generic named entity patterns
    GENERIC_PATTERNS = {
        "url": r"https?://[^\s<>\"]+",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "file_path": r"(?:/[\w.-]+)+(?:\.\w+)?",
        "version": r"\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?",
    }

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: Optional[Any] = None,
        domain_transformer: Optional[Any] = None,
    ) -> None:
        """
        Initialize the entity extractor.

        Args:
            use_llm: Whether to use LLM for extraction
            llm_provider: LLM provider instance
            domain_transformer: Domain transformer for type mapping
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.domain_transformer = domain_transformer

    def extract(
        self,
        content: str,
        content_type: str,
        source_path: str = "",
        language: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract entities from content.

        Args:
            content: Text content to extract from
            content_type: MIME type of the content
            source_path: Source path for context
            language: Programming language (if applicable)

        Returns:
            ExtractionResult with extracted entities
        """
        result = ExtractionResult(
            source_path=source_path,
            content_type=content_type,
        )

        # Determine extraction method based on content type
        if self._is_code_file(content_type):
            lang = language or self._detect_language(content_type, source_path)
            result = self._extract_from_code(content, lang, source_path)
        else:
            result = self._extract_from_text(content, source_path)

        # Use LLM for additional extraction if configured
        if self.use_llm and self.llm_provider:
            llm_entities = self._extract_with_llm(content)
            result.entities.extend(llm_entities)

        # Apply domain transformer if configured
        if self.domain_transformer:
            result.entities = [
                self.domain_transformer.map_entity(e) or e for e in result.entities
            ]

        # Deduplicate entities
        result.entities = self._deduplicate_entities(result.entities)

        return result

    def _is_code_file(self, content_type: str) -> bool:
        """Check if content type indicates a code file."""
        code_types = [
            "text/x-python",
            "application/javascript",
            "application/typescript",
            "text/x-java",
            "text/x-go",
            "text/x-rust",
            "text/x-c",
            "text/x-c++",
        ]
        return content_type in code_types

    def _detect_language(self, content_type: str, source_path: str) -> str:
        """Detect programming language from content type or path."""
        type_map = {
            "text/x-python": "python",
            "application/javascript": "javascript",
            "application/typescript": "javascript",
            "text/x-java": "java",
            "text/x-go": "go",
        }

        if content_type in type_map:
            return type_map[content_type]

        # Fallback to extension
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "javascript",
            ".jsx": "javascript",
            ".tsx": "javascript",
            ".java": "java",
            ".go": "go",
        }

        for ext, lang in ext_map.items():
            if source_path.endswith(ext):
                return lang

        return "unknown"

    def _extract_from_code(
        self,
        content: str,
        language: str,
        source_path: str,
    ) -> ExtractionResult:
        """Extract entities from code content."""
        result = ExtractionResult(
            source_path=source_path,
            content_type=f"text/x-{language}",
        )

        patterns = self.CODE_PATTERNS.get(language, {})

        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE)

            for match in matches:
                # Get the first non-None group
                name = next((g for g in match.groups() if g), None)
                if name:
                    # Handle import special case
                    if entity_type == "import" and "," in name:
                        # Split multiple imports
                        for imported in name.split(","):
                            imported = imported.strip()
                            if imported:
                                entity = self._create_entity(
                                    name=imported,
                                    entity_type="dependency",
                                    source_path=source_path,
                                    line_number=content[:match.start()].count("\n") + 1,
                                )
                                result.entities.append(entity)
                    else:
                        entity = self._create_entity(
                            name=name,
                            entity_type=entity_type,
                            source_path=source_path,
                            line_number=content[:match.start()].count("\n") + 1,
                        )
                        result.entities.append(entity)

                    result.raw_mentions.append({
                        "text": match.group(0),
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "line": content[:match.start()].count("\n") + 1,
                    })

        return result

    def _extract_from_text(
        self,
        content: str,
        source_path: str,
    ) -> ExtractionResult:
        """Extract entities from generic text content."""
        result = ExtractionResult(
            source_path=source_path,
            content_type="text/plain",
        )

        for entity_type, pattern in self.GENERIC_PATTERNS.items():
            matches = re.finditer(pattern, content)

            for match in matches:
                name = match.group(0)
                entity = self._create_entity(
                    name=name,
                    entity_type=entity_type,
                    source_path=source_path,
                )
                result.entities.append(entity)

                result.raw_mentions.append({
                    "text": name,
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                })

        return result

    def _extract_with_llm(self, content: str) -> List[Entity]:
        """Use LLM to extract entities from content."""
        if not self.llm_provider:
            return []

        try:
            extracted = self.llm_provider.extract_entities(content)
            # Convert ExtractedEntity to Entity
            entities = []
            for e in extracted:
                entity = Entity(
                    name=e.text,
                    entity_type=e.entity_type,
                    confidence=e.confidence,
                    properties={
                        "start_char": e.start_char,
                        "end_char": e.end_char,
                        "source_text": e.source_text,
                    },
                )
                entities.append(entity)
            return entities
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return []

    def _create_entity(
        self,
        name: str,
        entity_type: str,
        source_path: str = "",
        line_number: Optional[int] = None,
    ) -> Entity:
        """Create an Entity object."""
        properties = {"source_path": source_path}
        if line_number:
            properties["line_number"] = line_number

        return Entity(
            name=name,
            entity_type=entity_type,
            properties=properties,
        )

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on name and type."""
        seen = set()
        unique = []

        for entity in entities:
            key = (entity.name, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique


__all__ = ["EntityExtractor", "ExtractionResult"]
