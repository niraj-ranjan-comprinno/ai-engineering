"""
Document Loading and Chunking.

=============================================================================
THEORY: Why Chunking Matters
=============================================================================

LLMs have limited context windows (e.g., 128K tokens for GPT-4).
Even with large windows, stuffing everything in:
1. Costs more (pay per token)
2. Reduces quality (needle in haystack problem)
3. Slower (more to process)

SOLUTION: Split documents into chunks, retrieve only relevant ones.

Chunking Strategies:
--------------------
1. Fixed Size: Split every N characters/tokens
   - Simple but may cut mid-sentence
   
2. Recursive: Try to split by paragraphs, then sentences, then words
   - Better semantic boundaries
   
3. Semantic: Use embeddings to find natural break points
   - Best quality but more complex

Overlap:
--------
Without overlap, you might lose context:
    Chunk 1: "The Python programming language was created by..."
    Chunk 2: "...Guido van Rossum in 1991."
    
With overlap:
    Chunk 1: "The Python programming language was created by Guido van Rossum"
    Chunk 2: "created by Guido van Rossum in 1991."
    
The overlap ensures context isn't lost at boundaries.

=============================================================================
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a source document.
    
    THEORY: Document Metadata
    -------------------------
    Metadata is crucial for RAG systems:
    - source: Where did this come from? (for citations)
    - created_at: When was it indexed?
    - doc_type: PDF, markdown, etc. (for parsing)
    - Custom fields: author, category, version, etc.
    
    Good metadata enables:
    - Filtering (only search PDFs from 2024)
    - Attribution (cite the source)
    - Debugging (why was this retrieved?)
    """
    content: str
    source: str
    doc_type: str = "text"
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def doc_id(self) -> str:
        """Generate unique ID from content hash."""
        return hashlib.md5(f"{self.source}:{self.content[:100]}".encode()).hexdigest()


@dataclass
class Chunk:
    """
    A chunk of text from a document.
    
    Each chunk has:
    - content: The actual text
    - doc_id: Which document it came from
    - chunk_index: Position in the document
    - metadata: Inherited from document + chunk-specific
    """
    content: str
    doc_id: str
    chunk_index: int
    source: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        """Unique ID for this chunk."""
        return f"{self.doc_id}_{self.chunk_index}"


class TextChunker:
    """
    Splits documents into overlapping chunks.
    
    THEORY: Chunking Parameters
    ---------------------------
    
    chunk_size: 
        - Too small (100): Loses context, too many chunks
        - Too large (5000): Dilutes relevance, costs more
        - Sweet spot: 500-1500 characters
    
    chunk_overlap:
        - 0%: Risk losing context at boundaries
        - 50%: Too much redundancy, wastes storage
        - 10-20%: Good balance
    
    Example with chunk_size=1000, overlap=200:
    
    Document: [=====================================] 2500 chars
    
    Chunk 1:  [===========]                           0-1000
    Chunk 2:         [===========]                    800-1800
    Chunk 3:                [===========]             1600-2500
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks.
        
        Uses recursive splitting strategy:
        1. Try to split by paragraphs
        2. If paragraph too big, split by sentences
        3. If sentence too big, split by words
        """
        text = document.content.strip()
        
        if not text:
            return []
        
        # Split into chunks
        chunks = self._recursive_split(text)
        
        # Create Chunk objects
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                chunk_index=i,
                source=document.source,
                metadata={
                    **document.metadata,
                    "doc_type": document.doc_type,
                    "total_chunks": len(chunks),
                },
            )
            result.append(chunk)
        
        logger.info(f"Document '{document.source}' split into {len(result)} chunks")
        return result
    
    def _recursive_split(self, text: str) -> list[str]:
        """
        Recursively split text trying to maintain semantic boundaries.
        
        THEORY: Recursive Character Text Splitter
        -----------------------------------------
        This is the strategy used by LangChain and similar libraries.
        
        Separators in order of preference:
        1. Double newline (paragraphs)
        2. Single newline (lines)
        3. Period + space (sentences)
        4. Space (words)
        5. Empty string (characters - last resort)
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_with_separators(text, separators)
    
    def _split_with_separators(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Split text using separators in order of preference."""
        if not separators:
            # Last resort: split by character
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Use simple size-based splitting
            return self._split_by_size(text)
        
        # Split by current separator
        parts = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Add separator back (except for first part)
            part_with_sep = part if not current_chunk else separator + part
            
            if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                # Fits in current chunk
                current_chunk += part_with_sep
            else:
                # Doesn't fit
                if current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.strip())
                
                if len(part) > self.chunk_size:
                    # Part is too big, recursively split with next separator
                    sub_chunks = self._split_with_separators(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Start new chunk with this part
                    current_chunk = part
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Apply overlap
        return self._apply_overlap(chunks)
    
    def _split_by_size(self, text: str) -> list[str]:
        """Simple size-based splitting as last resort."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply overlap between chunks.
        
        Takes the last chunk_overlap characters from previous chunk
        and prepends to current chunk.
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get overlap text from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Only add overlap if it doesn't make chunk too big
            if len(overlap_text) + len(curr_chunk) <= self.chunk_size * 1.2:
                # Find a good break point in overlap (word boundary)
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                
                result.append(overlap_text + " " + curr_chunk)
            else:
                result.append(curr_chunk)
        
        return result


class DocumentLoader:
    """
    Load documents from various file formats.
    
    THEORY: Document Parsing
    ------------------------
    Different formats need different handling:
    
    - Text/Markdown: Simple, direct read
    - PDF: Complex, may have images, tables, multiple columns
    - DOCX: XML-based, preserves formatting
    - HTML: Need to strip tags, handle structure
    
    Challenges:
    - Encoding issues (UTF-8, Latin-1, etc.)
    - Malformed files
    - Large files (memory management)
    - Embedded content (images, tables)
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html"}
    
    def load_file(self, file_path: str | Path) -> Document:
        """Load a single file and return a Document."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        
        logger.info(f"Loading file: {path}")
        
        if extension == ".txt":
            content = self._load_text(path)
        elif extension == ".md":
            content = self._load_markdown(path)
        elif extension == ".pdf":
            content = self._load_pdf(path)
        elif extension == ".docx":
            content = self._load_docx(path)
        elif extension == ".html":
            content = self._load_html(path)
        else:
            content = self._load_text(path)
        
        return Document(
            content=content,
            source=str(path),
            doc_type=extension[1:],  # Remove the dot
            metadata={
                "filename": path.name,
                "file_size": path.stat().st_size,
            },
        )
    
    def load_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        """Load all supported files from a directory."""
        path = Path(dir_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {path}")
        return documents
    
    def _load_text(self, path: Path) -> str:
        """Load plain text file."""
        return path.read_text(encoding="utf-8")
    
    def _load_markdown(self, path: Path) -> str:
        """
        Load markdown file.
        
        For RAG, we keep the markdown as-is (don't convert to HTML).
        The structure (headers, lists) helps with context.
        """
        return path.read_text(encoding="utf-8")
    
    def _load_pdf(self, path: Path) -> str:
        """
        Load PDF file using pypdf.
        
        THEORY: PDF Extraction Challenges
        ---------------------------------
        PDFs are designed for display, not extraction:
        - Text may be in arbitrary order
        - Tables lose structure
        - Multi-column layouts confuse extractors
        - Scanned PDFs need OCR
        
        For production, consider:
        - Adobe PDF Extract API
        - Amazon Textract
        - Unstructured.io
        """
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(path)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")
    
    def _load_docx(self, path: Path) -> str:
        """Load DOCX file using python-docx."""
        try:
            from docx import Document as DocxDocument
            
            doc = DocxDocument(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
            
        except ImportError:
            raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")
    
    def _load_html(self, path: Path) -> str:
        """
        Load HTML file and extract text.
        
        Uses BeautifulSoup to parse and extract readable text.
        """
        try:
            from bs4 import BeautifulSoup
            
            html_content = path.read_text(encoding="utf-8")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator="\n")
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            return "\n".join(line for line in lines if line)
            
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML support. Install with: pip install beautifulsoup4")
