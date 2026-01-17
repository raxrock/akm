"""LangChain LLM provider implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from akm.core.config import LangChainConfig
from akm.core.exceptions import QueryError
from akm.core.interfaces import LLMProvider
from akm.core.models import ExtractedEntity, ExtractedRelationship, SearchResult

logger = logging.getLogger(__name__)


class LangChainLLMProvider(LLMProvider):
    """LLM provider using LangChain."""

    def __init__(self, config: LangChainConfig) -> None:
        """
        Initialize LangChain LLM provider.

        Args:
            config: LangChain configuration
        """
        self._config = config
        self._llm = None
        self._async_llm = None

    def _get_llm(self):
        """Get or create LangChain LLM."""
        if self._llm is not None:
            return self._llm

        try:
            provider = self._config.provider.lower()

            if provider == "openai":
                from langchain_openai import ChatOpenAI

                api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
                self._llm = ChatOpenAI(
                    model=self._config.model_name,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    api_key=api_key,
                )

            elif provider == "azure_openai" or provider == "azure":
                from langchain_openai import AzureChatOpenAI

                api_key = self._config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
                endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
                # Extract deployment name from model_name or use default
                deployment = self._config.model_name or "gpt-4"

                self._llm = AzureChatOpenAI(
                    azure_deployment=deployment,
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version="2024-02-15-preview",
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )

            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                api_key = self._config.api_key or os.environ.get("ANTHROPIC_API_KEY")
                self._llm = ChatAnthropic(
                    model=self._config.model_name,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    api_key=api_key,
                )

            elif provider == "ollama":
                from langchain_community.llms import Ollama

                self._llm = Ollama(
                    model=self._config.model_name,
                    temperature=self._config.temperature,
                )

            else:
                raise QueryError(f"Unsupported LLM provider: {provider}")

            return self._llm

        except ImportError as e:
            raise QueryError(
                f"Required LangChain package not installed: {e}. "
                f"Install with: pip install langchain-{self._config.provider}"
            )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)

            # Handle different response types
            if hasattr(response, "content"):
                return response.content
            return str(response)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise QueryError(f"Failed to generate response: {e}")

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """
        Async version of generate.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            llm = self._get_llm()
            response = await llm.ainvoke(prompt)

            if hasattr(response, "content"):
                return response.content
            return str(response)

        except Exception as e:
            logger.error(f"Async LLM generation failed: {e}")
            raise QueryError(f"Failed to generate response: {e}")

    def generate_with_context(
        self,
        prompt: str,
        context: List[SearchResult],
        **kwargs: Any,
    ) -> str:
        """
        Generate with additional context items.

        Args:
            prompt: User prompt/question
            context: List of context items
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        # Build context string
        context_parts = []
        for i, item in enumerate(context[:10], 1):
            context_parts.append(
                f"[{i}] {item.content}"
            )

        context_str = "\n\n".join(context_parts)

        # Build full prompt
        full_prompt = f"""Use the following context to answer the question.

Context:
{context_str}

Question: {prompt}

Answer based on the context provided. If the context doesn't contain relevant information, say so.

Answer:"""

        return self.generate(full_prompt, **kwargs)

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text using LLM.

        Args:
            text: Text to extract entities from
            entity_types: Optional list of entity types to extract

        Returns:
            List of extracted entities
        """
        types_str = ", ".join(entity_types) if entity_types else "any relevant entities"

        prompt = f"""Extract entities from the following text.
For each entity, provide:
- The exact text
- The entity type ({types_str})
- Start and end character positions
- Confidence score (0-1)

Text:
{text}

Return the entities in this format, one per line:
ENTITY|type|start|end|confidence

Only return the entities, no other text."""

        try:
            response = self.generate(prompt)
            entities = []

            for line in response.strip().split("\n"):
                line = line.strip()
                if not line or "|" not in line:
                    continue

                parts = line.split("|")
                if len(parts) >= 4:
                    try:
                        entities.append(
                            ExtractedEntity(
                                text=parts[0].strip(),
                                entity_type=parts[1].strip(),
                                start_char=int(parts[2].strip()),
                                end_char=int(parts[3].strip()),
                                confidence=float(parts[4].strip()) if len(parts) > 4 else 0.8,
                                source_text=text[:100],
                            )
                        )
                    except (ValueError, IndexError):
                        continue

            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        relationship_types: Optional[List[str]] = None,
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships from text using LLM.

        Args:
            text: Text to extract relationships from
            entities: Previously extracted entities (can be ExtractedEntity or strings)
            relationship_types: Optional list of relationship types

        Returns:
            List of extracted relationships
        """
        if not entities:
            return []

        # Handle both ExtractedEntity objects and plain strings
        def get_entity_text(e):
            if isinstance(e, str):
                return e
            return getattr(e, 'text', getattr(e, 'name', str(e)))

        def get_entity_type(e):
            if isinstance(e, str):
                return "entity"
            return getattr(e, 'entity_type', 'entity')

        def ensure_extracted_entity(e):
            """Convert to ExtractedEntity if needed."""
            if isinstance(e, ExtractedEntity):
                return e
            if isinstance(e, str):
                return ExtractedEntity(
                    text=e,
                    entity_type="entity",
                    start_char=0,
                    end_char=len(e),
                    confidence=0.8,
                )
            # Handle Entity or other objects with name attribute
            name = getattr(e, 'name', getattr(e, 'text', str(e)))
            etype = getattr(e, 'entity_type', 'entity')
            return ExtractedEntity(
                text=name,
                entity_type=etype,
                start_char=0,
                end_char=len(name),
                confidence=0.8,
            )

        entity_list = "\n".join(
            f"- {get_entity_text(e)} ({get_entity_type(e)})" for e in entities
        )
        types_str = (
            ", ".join(relationship_types)
            if relationship_types
            else "RELATED_TO, DEPENDS_ON, PART_OF, CREATED_BY, REFERENCES"
        )

        prompt = f"""Given these entities:
{entity_list}

And this text:
{text}

Extract relationships between the entities.
Relationship types: {types_str}

Return relationships in this format, one per line:
source_entity|relationship_type|target_entity|confidence

Only return the relationships, no other text."""

        try:
            response = self.generate(prompt)
            relationships = []

            # Build entity lookup (convert all to ExtractedEntity)
            entity_map = {
                get_entity_text(e).lower(): ensure_extracted_entity(e)
                for e in entities
            }

            for line in response.strip().split("\n"):
                line = line.strip()
                if not line or "|" not in line:
                    continue

                parts = line.split("|")
                if len(parts) >= 3:
                    source_text = parts[0].strip().lower()
                    rel_type = parts[1].strip().upper()
                    target_text = parts[2].strip().lower()

                    # Parse confidence, handling text values like "high", "medium", "low"
                    confidence = 0.7
                    if len(parts) > 3:
                        conf_str = parts[3].strip().lower()
                        try:
                            confidence = float(conf_str)
                        except ValueError:
                            # Handle text confidence values
                            confidence_map = {
                                "high": 0.9, "very high": 0.95,
                                "medium": 0.7, "moderate": 0.7,
                                "low": 0.4, "very low": 0.2,
                            }
                            confidence = confidence_map.get(conf_str, 0.7)

                    source_entity = entity_map.get(source_text)
                    target_entity = entity_map.get(target_text)

                    if source_entity and target_entity:
                        relationships.append(
                            ExtractedRelationship(
                                source_entity=source_entity,
                                target_entity=target_entity,
                                relationship_type=rel_type,
                                confidence=confidence,
                                source_text=text[:100],
                            )
                        )

            return relationships

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []


def create_langchain_provider(config: LangChainConfig) -> LangChainLLMProvider:
    """Factory function to create LangChain LLM provider."""
    return LangChainLLMProvider(config)
