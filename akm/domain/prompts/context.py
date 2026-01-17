"""Domain context injection for LLM prompts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List, Optional

from akm.core.models import DomainSchema, Entity, Relationship, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A template for generating prompts."""

    name: str
    template: str
    description: str = ""
    required_variables: List[str] = field(default_factory=list)
    default_values: Dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with given values.

        Args:
            **kwargs: Template variable values

        Returns:
            Rendered prompt string
        """
        # Merge defaults with provided values
        values = {**self.default_values, **kwargs}

        # Check required variables
        missing = [v for v in self.required_variables if v not in values]
        if missing:
            logger.warning(f"Missing template variables: {missing}")
            for var in missing:
                values[var] = f"[{var}]"

        # Use safe substitution
        try:
            tmpl = Template(self.template)
            return tmpl.safe_substitute(values)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            return self.template


@dataclass
class DomainContext:
    """Context information for a domain."""

    domain_name: str
    description: str = ""
    entity_types: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    terminology: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Convert to a string for prompt injection."""
        parts = []

        if self.description:
            parts.append(f"Domain: {self.domain_name}")
            parts.append(self.description)
            parts.append("")

        if self.entity_types:
            parts.append(f"Entity types: {', '.join(self.entity_types)}")

        if self.relationship_types:
            parts.append(f"Relationship types: {', '.join(self.relationship_types)}")

        if self.key_concepts:
            parts.append(f"Key concepts: {', '.join(self.key_concepts)}")

        if self.terminology:
            parts.append("")
            parts.append("Domain terminology:")
            for term, definition in self.terminology.items():
                parts.append(f"  - {term}: {definition}")

        if self.constraints:
            parts.append("")
            parts.append("Constraints:")
            for constraint in self.constraints:
                parts.append(f"  - {constraint}")

        return "\n".join(parts)


class ContextBuilder:
    """
    Builds context strings for LLM prompts.

    The context builder:
    1. Incorporates domain-specific information
    2. Formats entities and relationships
    3. Structures search results
    4. Manages context length limits
    """

    def __init__(
        self,
        schema: Optional[DomainSchema] = None,
        max_context_length: int = 8000,
    ) -> None:
        """
        Initialize the context builder.

        Args:
            schema: Optional domain schema
            max_context_length: Maximum context length in characters
        """
        self._schema = schema
        self._max_length = max_context_length
        self._domain_context: Optional[DomainContext] = None

        if schema:
            self._build_domain_context(schema)

    def _build_domain_context(self, schema: DomainSchema) -> None:
        """Build domain context from schema."""
        self._domain_context = DomainContext(
            domain_name=schema.domain_name,
            description=schema.description or "",
            entity_types=[t.name for t in schema.entity_types],
            relationship_types=[t.name for t in schema.relationship_types],
        )

        # Extract key concepts from entity descriptions
        for type_def in schema.entity_types:
            if type_def.description:
                self._domain_context.key_concepts.append(type_def.name)

    def build_query_context(
        self,
        query: str,
        search_results: Optional[List[SearchResult]] = None,
        entities: Optional[List[Entity]] = None,
        relationships: Optional[List[Relationship]] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Build context for a query.

        Args:
            query: User query
            search_results: Optional search results
            entities: Optional relevant entities
            relationships: Optional relevant relationships
            additional_context: Optional additional context string

        Returns:
            Formatted context string
        """
        parts = []
        current_length = 0

        # Add domain context
        if self._domain_context:
            domain_str = self._domain_context.to_context_string()
            if current_length + len(domain_str) < self._max_length:
                parts.append(domain_str)
                current_length += len(domain_str)

        # Add additional context
        if additional_context:
            if current_length + len(additional_context) < self._max_length:
                parts.append("")
                parts.append(additional_context)
                current_length += len(additional_context)

        # Add search results
        if search_results:
            search_context = self._format_search_results(
                search_results,
                max_length=self._max_length - current_length - 500,
            )
            parts.append("")
            parts.append("Relevant documents:")
            parts.append(search_context)
            current_length += len(search_context)

        # Add entity context
        if entities:
            entity_context = self._format_entities(
                entities,
                max_length=self._max_length - current_length - 300,
            )
            parts.append("")
            parts.append("Related entities:")
            parts.append(entity_context)
            current_length += len(entity_context)

        # Add relationship context
        if relationships and current_length < self._max_length - 200:
            rel_context = self._format_relationships(
                relationships,
                entities or [],
                max_length=self._max_length - current_length,
            )
            parts.append("")
            parts.append("Relationships:")
            parts.append(rel_context)

        return "\n".join(parts)

    def build_extraction_context(
        self,
        text: str,
        existing_entities: Optional[List[Entity]] = None,
    ) -> str:
        """
        Build context for entity/relationship extraction.

        Args:
            text: Text to extract from
            existing_entities: Optional existing entities for context

        Returns:
            Formatted context string
        """
        parts = []

        # Add domain context for extraction guidance
        if self._domain_context:
            parts.append("Domain context for extraction:")
            parts.append(f"Domain: {self._domain_context.domain_name}")

            if self._domain_context.entity_types:
                parts.append(f"Extract entities of types: {', '.join(self._domain_context.entity_types)}")

            if self._domain_context.relationship_types:
                parts.append(f"Identify relationships of types: {', '.join(self._domain_context.relationship_types)}")

            if self._domain_context.terminology:
                parts.append("")
                parts.append("Domain terminology to recognize:")
                for term, definition in list(self._domain_context.terminology.items())[:10]:
                    parts.append(f"  - {term}: {definition}")

        # Add existing entities for reference
        if existing_entities:
            parts.append("")
            parts.append("Known entities (for reference):")
            for entity in existing_entities[:10]:
                parts.append(f"  - {entity.name} ({entity.entity_type})")

        return "\n".join(parts)

    def _format_search_results(
        self,
        results: List[SearchResult],
        max_length: int,
    ) -> str:
        """Format search results for context."""
        parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            content = result.content
            # Truncate if needed
            if current_length + len(content) > max_length:
                remaining = max_length - current_length
                if remaining < 100:
                    break
                content = content[:remaining - 20] + "..."

            parts.append(f"[{i}] {content}")
            current_length += len(content)

            if current_length >= max_length:
                break

        return "\n\n".join(parts)

    def _format_entities(
        self,
        entities: List[Entity],
        max_length: int,
    ) -> str:
        """Format entities for context."""
        parts = []
        current_length = 0

        for entity in entities:
            entity_type = entity.entity_type
            if hasattr(entity_type, "value"):
                entity_type = entity_type.value

            line = f"- {entity.name} ({entity_type})"
            if entity.description:
                line += f": {entity.description[:100]}"

            if current_length + len(line) > max_length:
                break

            parts.append(line)
            current_length += len(line)

        return "\n".join(parts)

    def _format_relationships(
        self,
        relationships: List[Relationship],
        entities: List[Entity],
        max_length: int,
    ) -> str:
        """Format relationships for context."""
        parts = []
        current_length = 0

        # Build entity name lookup
        entity_names = {str(e.id): e.name for e in entities}

        for rel in relationships:
            source_name = entity_names.get(str(rel.source_id), str(rel.source_id)[:8])
            target_name = entity_names.get(str(rel.target_id), str(rel.target_id)[:8])

            rel_type = rel.relationship_type
            if hasattr(rel_type, "value"):
                rel_type = rel_type.value

            line = f"- {source_name} --[{rel_type}]--> {target_name}"

            if current_length + len(line) > max_length:
                break

            parts.append(line)
            current_length += len(line)

        return "\n".join(parts)


class PromptManager:
    """
    Manages prompt templates for different operations.

    Provides domain-aware prompts for:
    1. Entity extraction
    2. Relationship extraction
    3. Query answering
    4. Summarization
    """

    def __init__(
        self,
        schema: Optional[DomainSchema] = None,
    ) -> None:
        """
        Initialize the prompt manager.

        Args:
            schema: Optional domain schema
        """
        self._schema = schema
        self._templates: Dict[str, PromptTemplate] = {}
        self._context_builder = ContextBuilder(schema)

        # Register default templates
        self._register_default_templates()

        # Load schema-specific templates
        if schema:
            self._load_schema_templates(schema)

    def _register_default_templates(self) -> None:
        """Register default prompt templates."""
        self._templates["extraction"] = PromptTemplate(
            name="extraction",
            template="""Extract entities and relationships from the following text.

$domain_context

Text to analyze:
$text

Instructions:
1. Identify all named entities (people, organizations, concepts, documents, etc.)
2. Determine relationships between entities
3. Provide confidence scores for each extraction

Output format:
Entities:
- [name] | [type] | [description] | [confidence]

Relationships:
- [source] | [relationship_type] | [target] | [confidence]""",
            required_variables=["text"],
            default_values={"domain_context": ""},
        )

        self._templates["query_answer"] = PromptTemplate(
            name="query_answer",
            template="""Based on the following context, answer the question.

$domain_context

Context:
$context

Question: $question

Instructions:
1. Answer based only on the provided context
2. If information is insufficient, say so clearly
3. Cite relevant sources when possible
4. Be concise but complete

Answer:""",
            required_variables=["context", "question"],
            default_values={"domain_context": ""},
        )

        self._templates["summarization"] = PromptTemplate(
            name="summarization",
            template="""Summarize the following content.

$domain_context

Content:
$content

Instructions:
1. Provide a concise summary (2-3 paragraphs)
2. Highlight key entities and relationships
3. Note any important decisions or outcomes

Summary:""",
            required_variables=["content"],
            default_values={"domain_context": ""},
        )

        self._templates["relationship_extraction"] = PromptTemplate(
            name="relationship_extraction",
            template="""Given the following entities, identify relationships between them based on the text.

$domain_context

Known entities:
$entities

Text:
$text

Instructions:
1. Look for explicit and implicit relationships
2. Use relationship types: $relationship_types
3. Provide confidence scores (0-1)

Relationships found:""",
            required_variables=["entities", "text"],
            default_values={
                "domain_context": "",
                "relationship_types": "RELATED_TO, DEPENDS_ON, CREATED_BY, PART_OF, USES",
            },
        )

    def _load_schema_templates(self, schema: DomainSchema) -> None:
        """Load templates from schema."""
        if schema.extraction_prompt_template:
            self._templates["extraction"] = PromptTemplate(
                name="extraction",
                template=schema.extraction_prompt_template,
                required_variables=["text"],
            )

        if schema.query_context_template:
            self._templates["query_context"] = PromptTemplate(
                name="query_context",
                template=schema.query_context_template,
                required_variables=[],
            )

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def render_extraction_prompt(
        self,
        text: str,
        existing_entities: Optional[List[Entity]] = None,
    ) -> str:
        """Render an extraction prompt."""
        template = self._templates.get("extraction")
        if not template:
            return f"Extract entities and relationships from: {text}"

        domain_context = self._context_builder.build_extraction_context(
            text, existing_entities
        )

        return template.render(
            text=text,
            domain_context=domain_context,
        )

    def render_query_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """Render a query answering prompt."""
        template = self._templates.get("query_answer")
        if not template:
            return f"Question: {question}\n\nContext: {context}\n\nAnswer:"

        domain_context = ""
        if self._schema:
            domain_context = f"Domain: {self._schema.domain_name}"
            if self._schema.description:
                domain_context += f"\n{self._schema.description}"

        return template.render(
            question=question,
            context=context,
            domain_context=domain_context,
        )

    def render_summarization_prompt(
        self,
        content: str,
    ) -> str:
        """Render a summarization prompt."""
        template = self._templates.get("summarization")
        if not template:
            return f"Summarize the following:\n\n{content}"

        domain_context = ""
        if self._schema:
            domain_context = f"Domain: {self._schema.domain_name}"

        return template.render(
            content=content,
            domain_context=domain_context,
        )

    def render_relationship_prompt(
        self,
        text: str,
        entities: List[Entity],
    ) -> str:
        """Render a relationship extraction prompt."""
        template = self._templates.get("relationship_extraction")
        if not template:
            entity_names = ", ".join(e.name for e in entities)
            return f"Find relationships between: {entity_names}\n\nText: {text}"

        # Format entities
        entity_lines = []
        for entity in entities:
            entity_type = entity.entity_type
            if hasattr(entity_type, "value"):
                entity_type = entity_type.value
            entity_lines.append(f"- {entity.name} ({entity_type})")

        # Get relationship types
        rel_types = "RELATED_TO, DEPENDS_ON, CREATED_BY, PART_OF, USES"
        if self._schema:
            rel_types = ", ".join(t.name for t in self._schema.relationship_types) or rel_types

        domain_context = ""
        if self._schema:
            domain_context = f"Domain: {self._schema.domain_name}"

        return template.render(
            text=text,
            entities="\n".join(entity_lines),
            relationship_types=rel_types,
            domain_context=domain_context,
        )

    def register_template(self, template: PromptTemplate) -> None:
        """Register a custom template."""
        self._templates[template.name] = template

    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self._templates.keys())


def build_domain_context_string(
    schema: DomainSchema,
    include_types: bool = True,
    include_terminology: bool = False,
) -> str:
    """
    Build a domain context string from schema.

    Args:
        schema: Domain schema
        include_types: Include entity/relationship types
        include_terminology: Include domain terminology

    Returns:
        Formatted context string
    """
    parts = [f"Domain: {schema.domain_name}"]

    if schema.description:
        parts.append(schema.description)

    if include_types:
        if schema.entity_types:
            type_names = [t.name for t in schema.entity_types]
            parts.append(f"Entity types: {', '.join(type_names)}")

        if schema.relationship_types:
            rel_names = [t.name for t in schema.relationship_types]
            parts.append(f"Relationship types: {', '.join(rel_names)}")

    return "\n".join(parts)
