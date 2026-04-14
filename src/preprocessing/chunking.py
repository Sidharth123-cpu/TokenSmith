import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------- Chunking Configs --------------------------

class ChunkConfig(ABC):
    @abstractmethod
    def validate(self):
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        pass

@dataclass
class SectionRecursiveConfig(ChunkConfig):
    """Configuration for section-based chunking with recursive splitting."""
    recursive_chunk_size: int
    recursive_overlap: int
    
    def to_string(self) -> str:
        return f"chunk_mode=sections+recursive, chunk_size={self.recursive_chunk_size}, overlap={self.recursive_overlap}"

    def validate(self):
        assert self.recursive_chunk_size > 0, "recursive_chunk_size must be > 0"
        assert self.recursive_overlap >= 0, "recursive_overlap must be >= 0"


@dataclass
class SlidingWindowConfig(ChunkConfig):
    window_size: int
    overlap: int

    def to_string(self) -> str:
        return f"chunk_mode=sliding_window, window_size={self.window_size}, overlap={self.overlap}"

    def validate(self):
        assert self.window_size > 0, "window_size must be > 0"
        assert 0 <= self.overlap < self.window_size, "overlap must be >= 0 and < window_size"


@dataclass
class SentenceBoundaryConfig(ChunkConfig):
    max_chunk_size: int
    overlap_sentences: int = 1

    def to_string(self) -> str:
        return f"chunk_mode=sentence_boundary, max_chunk_size={self.max_chunk_size}, overlap_sentences={self.overlap_sentences}"

    def validate(self):
        assert self.max_chunk_size > 0, "max_chunk_size must be > 0"
        assert self.overlap_sentences >= 0, "overlap_sentences must be >= 0"


@dataclass
class ParagraphAwareConfig(ChunkConfig):
    max_chunk_size: int
    overlap: int = 0

    def to_string(self) -> str:
        return f"chunk_mode=paragraph, max_chunk_size={self.max_chunk_size}, overlap={self.overlap}"

    def validate(self):
        assert self.max_chunk_size > 0, "max_chunk_size must be > 0"
        assert self.overlap >= 0, "overlap must be >= 0"


@dataclass
class AdaptiveConfig(ChunkConfig):
    max_chunk_size: int
    overlap: int = 200

    def to_string(self) -> str:
        return f"chunk_mode=adaptive, max_chunk_size={self.max_chunk_size}, overlap={self.overlap}"

    def validate(self):
        assert self.max_chunk_size > 0, "max_chunk_size must be > 0"
        assert self.overlap >= 0, "overlap must be >= 0"


# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass
    
    @abstractmethod
    def artifact_folder_name(self) -> str:
        pass

class SectionRecursiveStrategy(ChunkStrategy):
    def __init__(self, config: SectionRecursiveConfig):
        self.config = config
        self.recursive_chunk_size = config.recursive_chunk_size
        self.recursive_overlap = config.recursive_overlap

    def name(self) -> str:
        return f"sections+recursive({self.recursive_chunk_size},{self.recursive_overlap})"

    def artifact_folder_name(self) -> str:
        return "sections"

    def chunk(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.recursive_chunk_size,
            chunk_overlap=self.recursive_overlap,
            separators=[". "]
        )
        return splitter.split_text(text)


class SlidingWindowStrategy(ChunkStrategy):
    def __init__(self, config: SlidingWindowConfig):
        self.config = config
        self.window_size = config.window_size
        self.overlap = config.overlap

    def name(self) -> str:
        return f"sliding_window({self.window_size},{self.overlap})"

    def artifact_folder_name(self) -> str:
        return "sliding_window"

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        step = self.window_size - self.overlap
        start = 0

        while start < len(text):
            end = start + self.window_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks


class SentenceBoundaryStrategy(ChunkStrategy):
    SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, config: SentenceBoundaryConfig):
        self.config = config
        self.max_chunk_size = config.max_chunk_size
        self.overlap_sentences = config.overlap_sentences

    def name(self) -> str:
        return f"sentence_boundary({self.max_chunk_size},{self.overlap_sentences})"

    def artifact_folder_name(self) -> str:
        return "sentence_boundary"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = self.SENTENCE_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return [text]

        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if sentence_len > self.max_chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_length = 0
                chunks.append(sentence)
                continue

            if current_length + sentence_len + (1 if current_sentences else 0) > self.max_chunk_size:
                chunks.append(" ".join(current_sentences))

                if self.overlap_sentences > 0 and len(current_sentences) > self.overlap_sentences:
                    current_sentences = current_sentences[-self.overlap_sentences:]
                    current_length = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_len + (1 if len(current_sentences) > 1 else 0)

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks


class ParagraphAwareStrategy(ChunkStrategy):
    PARAGRAPH_RE = re.compile(r'\n\s*\n')

    def __init__(self, config: ParagraphAwareConfig):
        self.config = config
        self.max_chunk_size = config.max_chunk_size
        self.overlap = config.overlap

    def name(self) -> str:
        return f"paragraph({self.max_chunk_size},{self.overlap})"

    def artifact_folder_name(self) -> str:
        return "paragraph"

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = self.PARAGRAPH_RE.split(text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return [text]

        chunks = []
        current_buffer = []
        current_length = 0

        sentence_config = SentenceBoundaryConfig(
            max_chunk_size=self.max_chunk_size,
            overlap_sentences=0
        )
        sentence_strategy = SentenceBoundaryStrategy(sentence_config)

        for para in paragraphs:
            para_len = len(para)

            if para_len > self.max_chunk_size:
                if current_buffer:
                    chunks.append("\n\n".join(current_buffer))
                    current_buffer = []
                    current_length = 0
                sub_chunks = sentence_strategy.chunk(para)
                chunks.extend(sub_chunks)
                continue

            separator_len = 2 if current_buffer else 0
            if current_length + para_len + separator_len > self.max_chunk_size:
                chunks.append("\n\n".join(current_buffer))
                current_buffer = []
                current_length = 0

            current_buffer.append(para)
            current_length += para_len + (2 if len(current_buffer) > 1 else 0)

        if current_buffer:
            chunks.append("\n\n".join(current_buffer))

        return chunks


class AdaptiveStrategy(ChunkStrategy):

    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.max_chunk_size = config.max_chunk_size
        self.overlap = config.overlap

    def name(self) -> str:
        return f"adaptive({self.max_chunk_size},{self.overlap})"

    def artifact_folder_name(self) -> str:
        return "adaptive"

    def _analyze_document(self, text: str) -> dict:
        paragraphs = ParagraphAwareStrategy.PARAGRAPH_RE.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = SentenceBoundaryStrategy.SENTENCE_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        total_len = len(text) if text else 1
        num_paragraphs = len(paragraphs)
        num_sentences = len(sentences)
        avg_paragraph_len = total_len / max(num_paragraphs, 1)
        avg_sentence_len = total_len / max(num_sentences, 1)
        paragraph_density = num_paragraphs / (total_len / 1000)
        has_headers = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))

        return {
            "num_paragraphs": num_paragraphs,
            "num_sentences": num_sentences,
            "avg_paragraph_len": avg_paragraph_len,
            "avg_sentence_len": avg_sentence_len,
            "paragraph_density": paragraph_density,
            "has_headers": has_headers,
            "total_length": total_len,
        }

    def _select_strategy(self, features: dict) -> ChunkStrategy:
        if features["paragraph_density"] > 2.0 or features["has_headers"]:
            config = ParagraphAwareConfig(
                max_chunk_size=self.max_chunk_size,
                overlap=self.overlap
            )
            return ParagraphAwareStrategy(config)

        if features["num_sentences"] > 5 and features["avg_sentence_len"] < 500:
            config = SentenceBoundaryConfig(
                max_chunk_size=self.max_chunk_size,
                overlap_sentences=1
            )
            return SentenceBoundaryStrategy(config)

        config = SlidingWindowConfig(
            window_size=self.max_chunk_size,
            overlap=self.overlap
        )
        return SlidingWindowStrategy(config)

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        features = self._analyze_document(text)
        strategy = self._select_strategy(features)
        return strategy.chunk(text)


# ----------------------------- Document Chunker ---------------------------------

class DocumentChunker:
    """
    Chunk text via a provided strategy.
    Table blocks (<table>...</table>) are preserved within chunks.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True
    ):
        self.strategy = strategy
        self.keep_tables = keep_tables

    def _extract_tables(self, text: str) -> Tuple[str, List[str]]:
        tables = self.TABLE_RE.findall(text)
        for i, t in enumerate(tables):
            text = text.replace(t, f"[TABLE_PLACEHOLDER_{i}]")
        return text, tables

    @staticmethod
    def _restore_tables(chunk: str, tables: List[str]) -> str:
        for i, t in enumerate(tables):
            ph = f"[TABLE_PLACEHOLDER_{i}]"
            if ph in chunk:
                chunk = chunk.replace(ph, t)
        return chunk

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        work = text
        tables: List[str] = []
        if self.keep_tables:
            work, tables = self._extract_tables(work)

        if self.strategy is None:
            raise ValueError("No chunk strategy provided")
        else:
            chunks = self.strategy.chunk(work)

        if self.keep_tables and tables:
            chunks = [self._restore_tables(c, tables) for c in chunks]
        return chunks
