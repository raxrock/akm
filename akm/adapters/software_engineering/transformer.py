"""Software Engineering Domain Transformer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from akm.core.interfaces import EmbeddingModel
from akm.core.models import DomainSchema, Entity, EntityType, Relationship
from akm.domain.transformer import BaseDomainTransformer

logger = logging.getLogger(__name__)


class SoftwareEngineeringTransformer(BaseDomainTransformer):
    """Domain transformer for software engineering contexts."""

    def __init__(self, schema_path: Optional[str] = None) -> None:
        """
        Initialize software engineering transformer.

        Args:
            schema_path: Optional path to custom schema file
        """
        if schema_path is None:
            schema_path = str(Path(__file__).parent / "schema.yaml")

        super().__init__(schema_path=schema_path)

    @property
    def domain_name(self) -> str:
        return "software_engineering"

    def map_entity(self, generic_entity: Entity) -> Optional[Entity]:
        """Map a generic entity to a software engineering entity."""
        # First apply base mapping
        mapped = super().map_entity(generic_entity)
        if not mapped:
            return generic_entity

        # Apply software engineering specific transformations
        mapped = self._apply_se_transformations(mapped)

        return mapped

    def _apply_se_transformations(self, entity: Entity) -> Entity:
        """Apply software engineering specific transformations."""
        domain_type = entity.domain_type

        if domain_type == "CodeFile":
            # Infer language from extension if not set
            if "language" not in entity.properties and "path" in entity.properties:
                path = entity.properties["path"]
                entity.properties["language"] = self._infer_language(path)

        elif domain_type == "Function":
            # Extract parameter count if signature is available
            if "signature" in entity.properties:
                sig = entity.properties["signature"]
                entity.properties["param_count"] = sig.count(",") + 1 if "(" in sig else 0

        return entity

    def _infer_language(self, file_path: str) -> str:
        """Infer programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".sh": "shell",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
        }

        path = Path(file_path)
        ext = path.suffix.lower()
        return ext_map.get(ext, "unknown")

    def get_prompt_context(self, query: str) -> str:
        """Get software engineering context for LLM prompts."""
        base_context = super().get_prompt_context(query)

        # Detect query intent and add specific context
        intent = self._detect_query_intent(query)

        additional_context = self._get_intent_context(intent)

        return f"{base_context}\n\n{additional_context}"

    def _detect_query_intent(self, query: str) -> str:
        """Detect the intent of a query."""
        query_lower = query.lower()

        intent_keywords = {
            "architecture": ["architecture", "design", "pattern", "structure", "module"],
            "debugging": ["bug", "error", "fix", "debug", "issue", "crash", "fail"],
            "code_review": ["review", "improve", "refactor", "clean", "quality"],
            "attribution": ["who", "author", "wrote", "created", "contributor"],
            "dependencies": ["depend", "import", "use", "library", "package", "version"],
            "api": ["api", "endpoint", "request", "response", "rest", "graphql"],
            "testing": ["test", "coverage", "unit", "integration", "mock"],
            "performance": ["performance", "slow", "optimize", "memory", "cpu"],
        }

        for intent, keywords in intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return intent

        return "general"

    def _get_intent_context(self, intent: str) -> str:
        """Get context specific to query intent."""
        intent_contexts = {
            "architecture": """
Focus on architectural patterns, design decisions, and their trade-offs.
Consider maintainability, scalability, and code organization.
Look for patterns like MVC, microservices, event-driven, etc.
""",
            "debugging": """
Focus on code flow, dependencies, and potential error sources.
Consider related issues and past bug fixes.
Look for error handling patterns and edge cases.
""",
            "code_review": """
Focus on code quality, best practices, and potential improvements.
Consider similar patterns in the codebase.
Look for opportunities to reduce complexity and improve readability.
""",
            "attribution": """
Focus on authorship, contributions, and expertise areas.
Consider commit history and code ownership.
Look for the most knowledgeable developers for specific areas.
""",
            "dependencies": """
Focus on package dependencies, versioning, and compatibility.
Consider security vulnerabilities and update recommendations.
Look for unused or outdated dependencies.
""",
            "api": """
Focus on API design, endpoints, and contracts.
Consider REST best practices, error handling, and documentation.
Look for consistency across API endpoints.
""",
            "testing": """
Focus on test coverage, test patterns, and testing strategies.
Consider unit tests, integration tests, and E2E tests.
Look for areas with insufficient test coverage.
""",
            "performance": """
Focus on performance bottlenecks and optimization opportunities.
Consider algorithmic complexity, caching, and resource usage.
Look for profiling data and performance metrics.
""",
        }

        return intent_contexts.get(intent, "")

    def validate_entity(self, entity: Entity) -> bool:
        """Validate an entity against the software engineering schema."""
        # First apply base validation
        if not super().validate_entity(entity):
            return False

        # Apply additional software engineering validations
        domain_type = entity.domain_type

        if domain_type == "CodeFile":
            # Must have a valid path
            path = entity.properties.get("path", "")
            if not path or not any(
                path.endswith(ext) for ext in [".py", ".js", ".ts", ".java", ".go", ".rs"]
            ):
                logger.warning(f"CodeFile entity has invalid path: {path}")
                # Not a hard failure, just a warning

        elif domain_type == "Repository":
            # Should have a URL
            if "url" not in entity.properties:
                logger.warning(f"Repository entity missing URL: {entity.name}")

        return True

    def extract_code_entities(
        self,
        code: str,
        language: str,
        file_path: Optional[str] = None,
    ) -> List[Entity]:
        """
        Extract entities from source code.

        This is a simplified implementation. A full implementation would use
        AST parsing for each supported language.

        Args:
            code: Source code content
            language: Programming language
            file_path: Optional file path

        Returns:
            List of extracted entities
        """
        entities = []

        # Simple pattern-based extraction (placeholder for AST parsing)
        if language == "python":
            entities.extend(self._extract_python_entities(code, file_path))
        elif language in ("javascript", "typescript"):
            entities.extend(self._extract_js_entities(code, file_path))

        return entities

    def _extract_python_entities(
        self,
        code: str,
        file_path: Optional[str],
    ) -> List[Entity]:
        """Extract entities from Python code."""
        import re

        entities = []

        # Extract classes
        class_pattern = r"class\s+(\w+)(?:\(([^)]*)\))?:"
        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            inheritance = match.group(2) or ""
            entities.append(
                Entity(
                    name=class_name,
                    entity_type="concept",
                    domain_type="Class",
                    description=f"Python class: {class_name}",
                    properties={
                        "language": "python",
                        "inheritance": [i.strip() for i in inheritance.split(",") if i.strip()],
                    },
                )
            )

        # Extract functions
        func_pattern = r"def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\w+))?"
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            params = match.group(2) or ""
            return_type = match.group(3)

            entities.append(
                Entity(
                    name=func_name,
                    entity_type="concept",
                    domain_type="Function",
                    description=f"Python function: {func_name}",
                    properties={
                        "language": "python",
                        "signature": f"{func_name}({params})",
                        "parameters": [p.strip().split(":")[0].strip() for p in params.split(",") if p.strip()],
                        "return_type": return_type,
                    },
                )
            )

        # Extract imports as dependencies
        import_pattern = r"(?:from\s+(\S+)\s+)?import\s+(\S+)"
        for match in re.finditer(import_pattern, code):
            module = match.group(1) or match.group(2)
            if module and not module.startswith("."):
                # External dependency
                entities.append(
                    Entity(
                        name=module.split(".")[0],
                        entity_type="concept",
                        domain_type="Dependency",
                        description=f"Python dependency: {module}",
                        properties={
                            "package_name": module.split(".")[0],
                        },
                    )
                )

        return entities

    def _extract_js_entities(
        self,
        code: str,
        file_path: Optional[str],
    ) -> List[Entity]:
        """Extract entities from JavaScript/TypeScript code."""
        import re

        entities = []

        # Extract classes
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            extends = match.group(2)
            entities.append(
                Entity(
                    name=class_name,
                    entity_type="concept",
                    domain_type="Class",
                    description=f"JavaScript class: {class_name}",
                    properties={
                        "language": "javascript",
                        "inheritance": [extends] if extends else [],
                    },
                )
            )

        # Extract functions
        func_patterns = [
            r"function\s+(\w+)\s*\(([^)]*)\)",  # function declarations
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>",  # arrow functions
        ]
        for pattern in func_patterns:
            for match in re.finditer(pattern, code):
                func_name = match.group(1)
                params = match.group(2) or ""
                entities.append(
                    Entity(
                        name=func_name,
                        entity_type="concept",
                        domain_type="Function",
                        description=f"JavaScript function: {func_name}",
                        properties={
                            "language": "javascript",
                            "signature": f"{func_name}({params})",
                            "parameters": [p.strip().split(":")[0].strip() for p in params.split(",") if p.strip()],
                        },
                    )
                )

        return entities
