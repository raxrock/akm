"""Document chunking strategies for vector storage."""

from __future__ import annotations

import re
from typing import List
from uuid import uuid4

from akm.core.interfaces import ChunkingStrategy
from akm.core.models import Chunk, Document


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size character-based chunking with overlap."""

    @property
    def strategy_name(self) -> str:
        return "fixed_size"

    def chunk(
        self,
        document: Document,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Chunk]:
        """
        Chunk document into fixed-size pieces.

        Args:
            document: Document to chunk
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            List of document chunks
        """
        content = document.content
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to end at a sentence or word boundary
            if end < len(content):
                # Look for sentence boundary
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_sep = content[start:end].rfind(sep)
                    if last_sep > chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
                else:
                    # Fall back to word boundary
                    last_space = content[start:end].rfind(" ")
                    if last_space > chunk_size // 2:
                        end = start + last_space + 1

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(
                    Chunk(
                        id=uuid4(),
                        content=chunk_content,
                        document_id=document.id,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

            # Move to next chunk with overlap
            start = end - chunk_overlap if end < len(content) else len(content)

        return chunks


class SentenceChunking(ChunkingStrategy):
    """Sentence-based chunking that groups sentences up to a size limit."""

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def chunk(
        self,
        document: Document,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Chunk]:
        """
        Chunk document by sentences, grouping them up to chunk_size.

        Args:
            document: Document to chunk
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Number of sentences to overlap

        Returns:
            List of document chunks
        """
        content = document.content

        # Split into sentences
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0
        chunk_index = 0
        start_char = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Check if adding this sentence exceeds the limit
            if current_length + sentence_len > chunk_size and current_sentences:
                # Create chunk from current sentences
                chunk_content = " ".join(current_sentences)
                end_char = start_char + len(chunk_content)

                chunks.append(
                    Chunk(
                        id=uuid4(),
                        content=chunk_content,
                        document_id=document.id,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
                chunk_index += 1

                # Keep overlap sentences
                overlap_count = min(chunk_overlap, len(current_sentences))
                if overlap_count > 0:
                    overlap_sentences = current_sentences[-overlap_count:]
                    start_char = end_char - sum(len(s) + 1 for s in overlap_sentences)
                    current_sentences = overlap_sentences
                    current_length = sum(len(s) + 1 for s in overlap_sentences)
                else:
                    start_char = end_char
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_len + 1

        # Don't forget the last chunk
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    id=uuid4(),
                    content=chunk_content,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                )
            )

        return chunks


class ParagraphChunking(ChunkingStrategy):
    """Paragraph-based chunking that respects document structure."""

    @property
    def strategy_name(self) -> str:
        return "paragraph"

    def chunk(
        self,
        document: Document,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Chunk]:
        """
        Chunk document by paragraphs, merging small paragraphs.

        Args:
            document: Document to chunk
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Not used for paragraph chunking

        Returns:
            List of document chunks
        """
        content = document.content

        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_paragraphs = []
        current_length = 0
        chunk_index = 0
        start_char = 0

        for para in paragraphs:
            para_len = len(para)

            # If single paragraph is too large, split it
            if para_len > chunk_size:
                # Flush current paragraphs first
                if current_paragraphs:
                    chunk_content = "\n\n".join(current_paragraphs)
                    chunks.append(
                        Chunk(
                            id=uuid4(),
                            content=chunk_content,
                            document_id=document.id,
                            chunk_index=chunk_index,
                            start_char=start_char,
                            end_char=start_char + len(chunk_content),
                        )
                    )
                    chunk_index += 1
                    start_char += len(chunk_content) + 2
                    current_paragraphs = []
                    current_length = 0

                # Split large paragraph using fixed-size chunking
                fixed_chunker = FixedSizeChunking()
                temp_doc = Document(
                    content=para,
                    source=document.source,
                    source_type=document.source_type,
                )
                sub_chunks = fixed_chunker.chunk(temp_doc, chunk_size, chunk_overlap)

                for sub_chunk in sub_chunks:
                    chunks.append(
                        Chunk(
                            id=uuid4(),
                            content=sub_chunk.content,
                            document_id=document.id,
                            chunk_index=chunk_index,
                            start_char=start_char + sub_chunk.start_char,
                            end_char=start_char + sub_chunk.end_char,
                        )
                    )
                    chunk_index += 1
                start_char += para_len + 2
                continue

            # Check if adding this paragraph exceeds the limit
            if current_length + para_len > chunk_size and current_paragraphs:
                chunk_content = "\n\n".join(current_paragraphs)
                chunks.append(
                    Chunk(
                        id=uuid4(),
                        content=chunk_content,
                        document_id=document.id,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                    )
                )
                chunk_index += 1
                start_char += len(chunk_content) + 2
                current_paragraphs = []
                current_length = 0

            current_paragraphs.append(para)
            current_length += para_len + 2

        # Don't forget the last chunk
        if current_paragraphs:
            chunk_content = "\n\n".join(current_paragraphs)
            chunks.append(
                Chunk(
                    id=uuid4(),
                    content=chunk_content,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                )
            )

        return chunks


def get_chunking_strategy(name: str = "fixed_size") -> ChunkingStrategy:
    """
    Get a chunking strategy by name.

    Args:
        name: Strategy name ("fixed_size", "sentence", "paragraph")

    Returns:
        ChunkingStrategy instance
    """
    strategies = {
        "fixed_size": FixedSizeChunking,
        "sentence": SentenceChunking,
        "paragraph": ParagraphChunking,
    }

    strategy_class = strategies.get(name.lower())
    if not strategy_class:
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategy_class()
