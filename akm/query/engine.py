"""Query engine for the AKM framework."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from akm.core.config import AKMConfig
from akm.core.exceptions import QueryError
from akm.core.interfaces import (
    DomainTransformer,
    EmbeddingModel,
    GraphBackend,
    LLMProvider,
    VectorBackend,
)
from akm.core.models import (
    ContextItem,
    Entity,
    Link,
    QueryResult,
    Relationship,
    SearchResult,
)
from akm.links.manager import LinkManager

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine that combines graph traversal, semantic search, and LLM reasoning.

    The QueryEngine orchestrates:
    1. Semantic search in vector database
    2. Graph traversal to find related entities
    3. Adaptive link consideration for context
    4. LLM-based answer synthesis
    """

    def __init__(
        self,
        graph: GraphBackend,
        vector: Optional[VectorBackend] = None,
        link_manager: Optional[LinkManager] = None,
        llm_provider: Optional[LLMProvider] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[AKMConfig] = None,
    ) -> None:
        """
        Initialize the query engine.

        Args:
            graph: Graph backend for entity/relationship queries
            vector: Vector backend for semantic search
            link_manager: Link manager for adaptive links
            llm_provider: LLM provider for answer synthesis
            embedding_model: Embedding model for query encoding
            config: AKM configuration
        """
        self._graph = graph
        self._vector = vector
        self._link_manager = link_manager
        self._llm = llm_provider
        self._embedding = embedding_model
        self._config = config or AKMConfig()
        self._domain_transformer: Optional[DomainTransformer] = None

    def set_domain_transformer(self, transformer: DomainTransformer) -> None:
        """Set the domain transformer for context injection."""
        self._domain_transformer = transformer

    def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results
            search_type: "semantic", "keyword", or "hybrid"
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        if not self._vector or not self._embedding:
            logger.warning("Vector backend or embedding model not configured")
            return []

        try:
            # Generate query embedding
            query_embedding = self._embedding.embed_single(query)

            # Get collection name from config
            collection_name = self._config.vector.chromadb.collection_name

            if search_type == "hybrid":
                return self._vector.hybrid_search(
                    collection_name=collection_name,
                    query_text=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                )
            else:
                return self._vector.search(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Entity]:
        """
        Search for entities by semantic similarity.

        Args:
            query: Search query
            entity_types: Filter by entity types
            top_k: Number of results

        Returns:
            List of matching entities
        """
        # First, try semantic search in vector DB
        search_results = self.search(query, top_k=top_k * 2)

        # Get entity IDs from search results
        entity_ids = set()
        for result in search_results:
            if "entity_id" in result.metadata:
                entity_ids.add(result.metadata["entity_id"])

        # Fetch entities from graph
        entities = []
        for entity_id in entity_ids:
            entity = self._graph.get_entity(entity_id)
            if entity:
                # Filter by type if specified
                if entity_types:
                    entity_type = (
                        entity.entity_type.value
                        if hasattr(entity.entity_type, "value")
                        else entity.entity_type
                    )
                    if entity_type not in entity_types:
                        continue
                entities.append(entity)

        return entities[:top_k]

    def query(
        self,
        question: str,
        context_entities: Optional[List[str]] = None,
        max_hops: int = 2,
        include_reasoning: bool = True,
    ) -> QueryResult:
        """
        Execute a natural language query.

        This method:
        1. Performs semantic search for relevant documents
        2. Expands context via graph traversal
        3. Considers adaptive links
        4. Synthesizes answer using LLM

        Args:
            question: Natural language question
            context_entities: Optional entity IDs for context
            max_hops: Maximum graph traversal hops
            include_reasoning: Include reasoning path in result

        Returns:
            QueryResult with answer and sources
        """
        start_time = time.time()
        reasoning_path = []

        try:
            # Step 1: Semantic search
            reasoning_path.append("Performing semantic search...")
            search_results = self.search(question, top_k=10)
            reasoning_path.append(f"Found {len(search_results)} relevant documents")

            # Step 2: Entity extraction from context
            entities_involved: List[Entity] = []
            relationships_involved: List[Relationship] = []
            links_involved: List[Link] = []

            if context_entities:
                reasoning_path.append(f"Using {len(context_entities)} context entities")
                for entity_id in context_entities:
                    entity = self._graph.get_entity(entity_id)
                    if entity:
                        entities_involved.append(entity)

            # Step 3: Graph traversal from identified entities
            if entities_involved and max_hops > 0:
                reasoning_path.append(f"Traversing graph (max depth: {max_hops})")
                for entity in entities_involved[:3]:  # Limit traversal starts
                    traversal = self._graph.traverse(
                        str(entity.id),
                        depth=max_hops,
                    )
                    for e in traversal.entities:
                        if e not in entities_involved:
                            entities_involved.append(e)
                    relationships_involved.extend(traversal.relationships)

            # Step 4: Get adaptive links
            if self._link_manager and entities_involved:
                reasoning_path.append("Considering adaptive links...")
                entity_ids = [str(e.id) for e in entities_involved]
                links_involved = self._link_manager.get_links_in_subgraph(
                    entity_ids,
                    min_weight=0.3,
                )
                reasoning_path.append(f"Found {len(links_involved)} relevant links")

            # Step 5: Build context for LLM
            context_items = self._build_context(
                search_results,
                entities_involved,
                relationships_involved,
                links_involved,
            )

            # Step 6: Generate answer with LLM
            if self._llm:
                reasoning_path.append("Synthesizing answer with LLM...")
                answer = self._generate_answer(question, context_items)
            else:
                # Fallback: return context summary
                answer = self._summarize_context(question, context_items)
                reasoning_path.append("LLM not configured, returning context summary")

            execution_time = (time.time() - start_time) * 1000

            return QueryResult(
                answer=answer,
                sources=search_results,
                entities_involved=entities_involved,
                relationships_involved=relationships_involved,
                links_involved=links_involved,
                reasoning_path=reasoning_path if include_reasoning else [],
                confidence=self._calculate_confidence(search_results, links_involved),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise QueryError(f"Query execution failed: {e}", query=question)

    def query_temporal(
        self,
        question: str,
        time_range: Optional[tuple] = None,
        entity_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Execute a temporal/decision lineage query.

        This is useful for understanding how decisions evolved over time.

        Args:
            question: Natural language question
            time_range: Optional (start, end) datetime tuple
            entity_id: Optional entity to focus on

        Returns:
            QueryResult with temporal context
        """
        # For temporal queries, we focus on:
        # 1. Link weight history (strengthening/weakening over time)
        # 2. Entity/relationship creation/modification times
        # 3. Document timestamps

        reasoning_path = ["Analyzing temporal context..."]

        # Get entity history if specified
        entities_involved: List[Entity] = []
        if entity_id:
            entity = self._graph.get_entity(entity_id)
            if entity:
                entities_involved.append(entity)
                # Get related entities ordered by creation time
                neighbors = self._graph.get_neighbors(entity_id, depth=2)
                neighbors.sort(key=lambda e: e.created_at)
                entities_involved.extend(neighbors[:10])

        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            entities_involved = [
                e
                for e in entities_involved
                if start_time <= e.created_at <= end_time
            ]

        # Get link evolution
        links_involved: List[Link] = []
        if self._link_manager and entities_involved:
            entity_ids = [str(e.id) for e in entities_involved]
            links_involved = self._link_manager.get_links_in_subgraph(entity_ids)
            # Sort by validation count (more validations = more established)
            links_involved.sort(
                key=lambda l: l.weight.validation_count, reverse=True
            )

        # Build temporal answer
        answer = self._build_temporal_answer(
            question, entities_involved, links_involved
        )

        return QueryResult(
            answer=answer,
            sources=[],
            entities_involved=entities_involved,
            relationships_involved=[],
            links_involved=links_involved,
            reasoning_path=reasoning_path,
            confidence=0.7,
            execution_time_ms=0,
        )

    def _build_context(
        self,
        search_results: List[SearchResult],
        entities: List[Entity],
        relationships: List[Relationship],
        links: List[Link],
    ) -> List[ContextItem]:
        """Build context items for LLM."""
        context_items = []

        # Add search results
        for result in search_results[:5]:
            context_items.append(
                ContextItem(
                    content=result.content,
                    source=f"document:{result.document_id}",
                    relevance_score=result.score,
                )
            )

        # Add entity context
        for entity in entities[:10]:
            content = f"{entity.name}: {entity.description or 'No description'}"
            if entity.properties:
                content += f"\nProperties: {entity.properties}"
            context_items.append(
                ContextItem(
                    content=content,
                    source=f"entity:{entity.id}",
                    relevance_score=entity.confidence,
                    entity_context=[entity],
                )
            )

        # Add high-weight link context
        for link in sorted(links, key=lambda l: l.weight.value, reverse=True)[:5]:
            source_entity = next(
                (e for e in entities if str(e.id) == str(link.source_id)), None
            )
            target_entity = next(
                (e for e in entities if str(e.id) == str(link.target_id)), None
            )
            if source_entity and target_entity:
                content = (
                    f"Connection: {source_entity.name} -> {target_entity.name} "
                    f"(weight: {link.weight.value:.2f}, type: {link.link_type})"
                )
                context_items.append(
                    ContextItem(
                        content=content,
                        source=f"link:{link.id}",
                        relevance_score=link.weight.value,
                    )
                )

        return context_items

    def _generate_answer(
        self,
        question: str,
        context_items: List[ContextItem],
    ) -> str:
        """Generate answer using LLM."""
        if not self._llm:
            return self._summarize_context(question, context_items)

        # Build prompt
        context_text = "\n\n".join(
            f"[{item.source}] {item.content}" for item in context_items
        )

        # Add domain context if available
        domain_context = ""
        if self._domain_transformer:
            domain_context = self._domain_transformer.get_prompt_context(question)

        prompt = f"""Based on the following context, answer the question.

{domain_context}

Context:
{context_text}

Question: {question}

Provide a clear, concise answer based on the context. If the context doesn't contain enough information, say so.

Answer:"""

        try:
            return self._llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._summarize_context(question, context_items)

    def _summarize_context(
        self,
        question: str,
        context_items: List[ContextItem],
    ) -> str:
        """Summarize context when LLM is not available."""
        if not context_items:
            return "No relevant information found in the knowledge base."

        summary_parts = [f"Regarding your question: '{question}'", ""]
        summary_parts.append("Relevant information found:")

        for item in context_items[:5]:
            summary_parts.append(f"- {item.content[:200]}...")

        return "\n".join(summary_parts)

    def _build_temporal_answer(
        self,
        question: str,
        entities: List[Entity],
        links: List[Link],
    ) -> str:
        """Build answer for temporal queries."""
        if not entities:
            return "No temporal information found for this query."

        parts = [f"Temporal analysis for: '{question}'", ""]

        if entities:
            parts.append("Entity timeline:")
            for entity in entities[:5]:
                parts.append(
                    f"  - {entity.created_at.strftime('%Y-%m-%d')}: "
                    f"{entity.name} ({entity.entity_type})"
                )

        if links:
            parts.append("")
            parts.append("Established connections:")
            for link in links[:5]:
                status = "strong" if link.weight.value > 0.7 else "emerging"
                parts.append(
                    f"  - {link.link_type}: weight {link.weight.value:.2f} ({status})"
                )

        return "\n".join(parts)

    def _calculate_confidence(
        self,
        search_results: List[SearchResult],
        links: List[Link],
    ) -> float:
        """Calculate overall confidence in the answer."""
        if not search_results and not links:
            return 0.0

        # Average of top search scores and link weights
        search_confidence = 0.0
        if search_results:
            top_scores = [r.score for r in search_results[:3]]
            search_confidence = sum(top_scores) / len(top_scores)

        link_confidence = 0.0
        if links:
            top_weights = [l.weight.value for l in sorted(
                links, key=lambda l: l.weight.value, reverse=True
            )[:3]]
            link_confidence = sum(top_weights) / len(top_weights)

        # Weighted average
        if search_results and links:
            return 0.6 * search_confidence + 0.4 * link_confidence
        elif search_results:
            return search_confidence
        else:
            return link_confidence
