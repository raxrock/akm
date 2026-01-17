"""Result synthesis with reasoning paths for LLM-powered queries."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from akm.core.interfaces import LLMProvider
from akm.core.models import (
    ContextItem,
    Entity,
    Link,
    QueryResult,
    Relationship,
    SearchResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""

    step_number: int
    action: str  # "search", "traverse", "filter", "link", "synthesize"
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        """Convert step to readable string."""
        return f"Step {self.step_number}: [{self.action}] {self.description}"


@dataclass
class ReasoningPath:
    """Complete reasoning path from query to answer."""

    query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_confidence: float = 0.0

    def add_step(
        self,
        action: str,
        description: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        confidence: float = 1.0,
    ) -> ReasoningStep:
        """Add a step to the reasoning path."""
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            action=action,
            description=description,
            inputs=inputs or [],
            outputs=outputs or [],
            confidence=confidence,
        )
        self.steps.append(step)
        return step

    def get_summary(self) -> List[str]:
        """Get a summary of the reasoning path."""
        return [step.to_string() for step in self.steps]

    def calculate_confidence(self) -> float:
        """Calculate overall confidence from step confidences."""
        if not self.steps:
            return 0.0

        # Geometric mean of step confidences
        product = 1.0
        for step in self.steps:
            product *= step.confidence

        self.final_confidence = product ** (1 / len(self.steps))
        return self.final_confidence


@dataclass
class SynthesisContext:
    """Context for answer synthesis."""

    query: str
    search_results: List[SearchResult] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    domain_context: str = ""
    reasoning_path: Optional[ReasoningPath] = None


class AnswerSynthesizer:
    """
    Synthesizes answers from gathered context.

    The synthesizer:
    1. Organizes context from multiple sources
    2. Builds prompts for LLM generation
    3. Tracks reasoning paths
    4. Generates coherent answers with sources
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        max_context_items: int = 10,
        max_tokens: int = 4000,
    ) -> None:
        """
        Initialize the answer synthesizer.

        Args:
            llm_provider: LLM provider for generation
            max_context_items: Maximum context items to include
            max_tokens: Approximate max tokens for context
        """
        self._llm = llm_provider
        self.max_context_items = max_context_items
        self.max_tokens = max_tokens

    def synthesize(
        self,
        context: SynthesisContext,
        include_reasoning: bool = True,
    ) -> QueryResult:
        """
        Synthesize an answer from context.

        Args:
            context: Synthesis context with all gathered information
            include_reasoning: Whether to include reasoning path

        Returns:
            Query result with answer
        """
        # Initialize reasoning path
        reasoning_path = context.reasoning_path or ReasoningPath(query=context.query)

        # Build context items
        context_items = self._build_context_items(context, reasoning_path)

        # Generate answer
        if self._llm:
            answer = self._generate_with_llm(context.query, context_items, context.domain_context)
            reasoning_path.add_step(
                action="synthesize",
                description="Generated answer using LLM with gathered context",
                inputs=[f"{len(context_items)} context items"],
                outputs=["Final answer"],
                confidence=0.9,
            )
        else:
            answer = self._generate_fallback(context.query, context_items)
            reasoning_path.add_step(
                action="synthesize",
                description="Generated summary without LLM",
                confidence=0.7,
            )

        # Calculate final confidence
        confidence = reasoning_path.calculate_confidence()

        # Adjust confidence based on source quality
        if context.search_results:
            avg_search_score = sum(r.score for r in context.search_results) / len(context.search_results)
            confidence = confidence * 0.7 + avg_search_score * 0.3

        return QueryResult(
            answer=answer,
            sources=context.search_results,
            entities_involved=context.entities,
            relationships_involved=context.relationships,
            links_involved=context.links,
            reasoning_path=reasoning_path.get_summary() if include_reasoning else [],
            confidence=min(1.0, confidence),
        )

    def _build_context_items(
        self,
        context: SynthesisContext,
        reasoning_path: ReasoningPath,
    ) -> List[ContextItem]:
        """Build prioritized context items."""
        items = []
        total_chars = 0
        char_limit = self.max_tokens * 4  # Rough estimate

        # Add search results (highest priority)
        for result in context.search_results[:self.max_context_items]:
            if total_chars + len(result.content) > char_limit:
                break

            items.append(
                ContextItem(
                    content=result.content,
                    source=f"document:{result.document_id}",
                    relevance_score=result.score,
                )
            )
            total_chars += len(result.content)

        reasoning_path.add_step(
            action="filter",
            description=f"Selected {len(items)} document chunks based on relevance",
            outputs=[f"{len(items)} context items"],
            confidence=0.95,
        )

        # Add entity context
        entity_count = 0
        for entity in context.entities[:10]:
            content = f"Entity: {entity.name}"
            if entity.description:
                content += f" - {entity.description}"
            if entity.properties:
                props_str = ", ".join(f"{k}: {v}" for k, v in list(entity.properties.items())[:5])
                content += f" (Properties: {props_str})"

            if total_chars + len(content) > char_limit:
                break

            items.append(
                ContextItem(
                    content=content,
                    source=f"entity:{entity.id}",
                    relevance_score=entity.confidence,
                    entity_context=[entity],
                )
            )
            total_chars += len(content)
            entity_count += 1

        if entity_count > 0:
            reasoning_path.add_step(
                action="traverse",
                description=f"Included context from {entity_count} relevant entities",
                confidence=0.9,
            )

        # Add high-weight link context
        link_count = 0
        sorted_links = sorted(context.links, key=lambda l: l.weight.value, reverse=True)
        for link in sorted_links[:5]:
            source_entity = next((e for e in context.entities if str(e.id) == str(link.source_id)), None)
            target_entity = next((e for e in context.entities if str(e.id) == str(link.target_id)), None)

            if source_entity and target_entity:
                content = (
                    f"Connection: {source_entity.name} <-> {target_entity.name} "
                    f"(type: {link.link_type}, strength: {link.weight.value:.2f})"
                )

                if total_chars + len(content) > char_limit:
                    break

                items.append(
                    ContextItem(
                        content=content,
                        source=f"link:{link.id}",
                        relevance_score=link.weight.value,
                    )
                )
                total_chars += len(content)
                link_count += 1

        if link_count > 0:
            reasoning_path.add_step(
                action="link",
                description=f"Considered {link_count} adaptive links between entities",
                confidence=0.85,
            )

        return items

    def _generate_with_llm(
        self,
        query: str,
        context_items: List[ContextItem],
        domain_context: str = "",
    ) -> str:
        """Generate answer using LLM."""
        if not self._llm:
            return self._generate_fallback(query, context_items)

        # Build prompt
        context_text = self._format_context(context_items)

        prompt = f"""Based on the following context, answer the question.

{f"Domain Context: {domain_context}" if domain_context else ""}

Context:
{context_text}

Question: {query}

Instructions:
1. Answer based only on the provided context
2. If the context doesn't contain enough information, say so
3. Cite relevant sources when possible
4. Be concise but complete

Answer:"""

        try:
            return self._llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback(query, context_items)

    def _generate_fallback(
        self,
        query: str,
        context_items: List[ContextItem],
    ) -> str:
        """Generate fallback answer without LLM."""
        if not context_items:
            return "No relevant information found in the knowledge base."

        parts = [f"Regarding your question: '{query}'", ""]

        # Group by source type
        documents = [c for c in context_items if c.source.startswith("document:")]
        entities = [c for c in context_items if c.source.startswith("entity:")]
        links = [c for c in context_items if c.source.startswith("link:")]

        if documents:
            parts.append("Relevant documents found:")
            for doc in documents[:3]:
                parts.append(f"- {doc.content[:200]}...")
            parts.append("")

        if entities:
            parts.append("Related entities:")
            for ent in entities[:5]:
                parts.append(f"- {ent.content}")
            parts.append("")

        if links:
            parts.append("Knowledge connections:")
            for link in links[:3]:
                parts.append(f"- {link.content}")

        return "\n".join(parts)

    def _format_context(self, items: List[ContextItem]) -> str:
        """Format context items for prompt."""
        parts = []
        for i, item in enumerate(items, 1):
            source_label = item.source.split(":")[0].title()
            parts.append(f"[{source_label} {i}] {item.content}")
        return "\n\n".join(parts)


class ReasoningBuilder:
    """
    Builds reasoning paths for query processing.

    Used to track and explain the reasoning process
    from query to answer.
    """

    def __init__(self) -> None:
        """Initialize the reasoning builder."""
        self._current_path: Optional[ReasoningPath] = None

    def start_reasoning(self, query: str) -> ReasoningPath:
        """Start a new reasoning path."""
        self._current_path = ReasoningPath(query=query)
        return self._current_path

    def add_search_step(
        self,
        num_results: int,
        search_type: str = "semantic",
    ) -> Optional[ReasoningStep]:
        """Add a search step."""
        if not self._current_path:
            return None

        return self._current_path.add_step(
            action="search",
            description=f"Performed {search_type} search",
            outputs=[f"Found {num_results} relevant documents"],
            confidence=0.95 if num_results > 0 else 0.5,
        )

    def add_traversal_step(
        self,
        start_entities: int,
        found_entities: int,
        depth: int,
    ) -> Optional[ReasoningStep]:
        """Add a graph traversal step."""
        if not self._current_path:
            return None

        return self._current_path.add_step(
            action="traverse",
            description=f"Traversed graph from {start_entities} starting points to depth {depth}",
            outputs=[f"Discovered {found_entities} related entities"],
            confidence=0.9 if found_entities > 0 else 0.6,
        )

    def add_link_step(
        self,
        num_links: int,
        avg_weight: float,
    ) -> Optional[ReasoningStep]:
        """Add an adaptive link consideration step."""
        if not self._current_path:
            return None

        return self._current_path.add_step(
            action="link",
            description=f"Evaluated {num_links} adaptive links",
            outputs=[f"Average link strength: {avg_weight:.2f}"],
            confidence=avg_weight,
        )

    def add_filter_step(
        self,
        before: int,
        after: int,
        criteria: str,
    ) -> Optional[ReasoningStep]:
        """Add a filtering step."""
        if not self._current_path:
            return None

        return self._current_path.add_step(
            action="filter",
            description=f"Filtered results by {criteria}",
            inputs=[f"{before} items"],
            outputs=[f"{after} items after filtering"],
            confidence=0.95,
        )

    def add_synthesis_step(
        self,
        method: str = "LLM",
        confidence: float = 0.9,
    ) -> Optional[ReasoningStep]:
        """Add a synthesis step."""
        if not self._current_path:
            return None

        return self._current_path.add_step(
            action="synthesize",
            description=f"Synthesized answer using {method}",
            outputs=["Final answer"],
            confidence=confidence,
        )

    def get_path(self) -> Optional[ReasoningPath]:
        """Get the current reasoning path."""
        return self._current_path

    def get_summary(self) -> List[str]:
        """Get summary of reasoning steps."""
        if not self._current_path:
            return []
        return self._current_path.get_summary()


def synthesize_answer(
    query: str,
    search_results: List[SearchResult],
    entities: List[Entity],
    relationships: List[Relationship],
    links: List[Link],
    llm_provider: Optional[LLMProvider] = None,
    domain_context: str = "",
) -> QueryResult:
    """
    Convenience function to synthesize an answer.

    Args:
        query: User query
        search_results: Search results
        entities: Involved entities
        relationships: Involved relationships
        links: Involved links
        llm_provider: Optional LLM provider
        domain_context: Optional domain context

    Returns:
        Query result with synthesized answer
    """
    # Build reasoning path
    builder = ReasoningBuilder()
    reasoning_path = builder.start_reasoning(query)

    builder.add_search_step(len(search_results))

    if entities:
        builder.add_traversal_step(
            start_entities=min(3, len(entities)),
            found_entities=len(entities),
            depth=2,
        )

    if links:
        avg_weight = sum(l.weight.value for l in links) / len(links) if links else 0
        builder.add_link_step(len(links), avg_weight)

    # Create context
    context = SynthesisContext(
        query=query,
        search_results=search_results,
        entities=entities,
        relationships=relationships,
        links=links,
        domain_context=domain_context,
        reasoning_path=reasoning_path,
    )

    # Synthesize
    synthesizer = AnswerSynthesizer(llm_provider=llm_provider)
    return synthesizer.synthesize(context)
