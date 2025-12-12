"""
Text chunking strategies for RAG.
Handles splitting documents into manageable chunks with overlap.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    metadata: Dict
    start_char: int
    end_char: int


class TextChunker:
    """Chunks text documents for vector embedding."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to split on (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for markdown and general text
        if separators is None:
            self.separators = [
                "\n\n## ",  # H2 headers
                "\n\n### ",  # H3 headers
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentences
                " ",  # Words
                ""  # Characters
            ]
        else:
            self.separators = separators
    
    def chunk_document(self, document: Dict[str, any]) -> List[Chunk]:
        """
        Chunk a document based on its type.
        
        Args:
            document: Document dictionary from DocumentLoader
            
        Returns:
            List of Chunk objects
        """
        doc_type = document.get('type', 'markdown')
        
        if doc_type == 'markdown':
            return self._chunk_markdown(document)
        elif doc_type == 'powerpoint':
            return self._chunk_powerpoint(document)
        else:
            return self._chunk_text(document['content'], document['metadata'])
    
    def _chunk_markdown(self, document: Dict[str, any]) -> List[Chunk]:
        """Chunk markdown document, preserving structure."""
        content = document.get('raw_content', document['content'])
        metadata = document['metadata'].copy()
        
        chunks = []
        
        # Split by headers first to preserve semantic structure
        # Pattern matches: newline, then # header, then newline
        sections = re.split(r'\n(#{1,3}\s+.+?)\n', content)
        
        # If no headers found or only one section, process entire content
        if len(sections) <= 1:
            # No headers or single section - chunk the whole document
            return self._split_text(content, metadata, metadata.get('title', 'Document'))
        
        current_section = ""
        section_title = metadata.get('title', 'Introduction')
        chunk_idx = 0
        
        for i, section in enumerate(sections):
            if section.strip().startswith('#'):
                # This is a header - save current section if it exists
                if current_section.strip():
                    # Process current section before moving to new header
                    if len(current_section) > self.chunk_size:
                        section_chunks = self._split_text(current_section, metadata, section_title, chunk_idx)
                        chunks.extend(section_chunks)
                        chunk_idx += len(section_chunks)
                    else:
                        chunk = Chunk(
                            text=current_section.strip(),
                            chunk_id=f"{metadata['filename']}_chunk_{chunk_idx}",
                            metadata={
                                **metadata,
                                'section_title': section_title,
                                'chunk_index': chunk_idx
                            },
                            start_char=0,
                            end_char=len(current_section)
                        )
                        chunks.append(chunk)
                        chunk_idx += 1
                    current_section = ""
                
                # Update section title
                section_title = section.strip()
                continue
            
            # Add section to current content
            if current_section:
                current_section += "\n\n" + section
            else:
                current_section = section
            
            # If section is too large, split it further
            if len(current_section) > self.chunk_size:
                # Split current section
                section_chunks = self._split_text(current_section, metadata, section_title, chunk_idx)
                chunks.extend(section_chunks)
                chunk_idx += len(section_chunks)
                current_section = ""
            elif len(current_section) >= self.chunk_size * 0.7:  # 70% of target size
                # Create chunk from current section
                chunk = Chunk(
                    text=current_section.strip(),
                    chunk_id=f"{metadata['filename']}_chunk_{chunk_idx}",
                    metadata={
                        **metadata,
                        'section_title': section_title,
                        'chunk_index': chunk_idx
                    },
                    start_char=0,  # Simplified
                    end_char=len(current_section)
                )
                chunks.append(chunk)
                chunk_idx += 1
                current_section = ""
        
        # Add remaining content
        if current_section.strip():
            if len(current_section) > self.chunk_size:
                section_chunks = self._split_text(current_section, metadata, section_title, chunk_idx)
                chunks.extend(section_chunks)
            else:
                chunk = Chunk(
                    text=current_section.strip(),
                    chunk_id=f"{metadata['filename']}_chunk_{chunk_idx}",
                    metadata={
                        **metadata,
                        'section_title': section_title,
                        'chunk_index': chunk_idx
                    },
                    start_char=0,
                    end_char=len(current_section)
                )
                chunks.append(chunk)
        
        return chunks if chunks else self._split_text(content, metadata, metadata.get('title', 'Document'))
    
    def _chunk_powerpoint(self, document: Dict[str, any]) -> List[Chunk]:
        """Chunk PowerPoint document by slides, with optional further splitting."""
        slides = document.get('slides', [])
        metadata = document['metadata'].copy()
        chunks = []
        
        # Strategy: Each slide is a chunk, but if slide is too large, split it
        for slide_idx, slide_content in enumerate(slides):
            slide_meta = metadata['slide_metadata'][slide_idx] if slide_idx < len(metadata.get('slide_metadata', [])) else {}
            
            if len(slide_content) <= self.chunk_size:
                # Slide fits in one chunk
                chunk = Chunk(
                    text=slide_content,
                    chunk_id=f"{metadata['filename']}_slide_{slide_idx + 1}",
                    metadata={
                        **metadata,
                        'slide_number': slide_idx + 1,
                        'slide_title': slide_meta.get('title', f"Slide {slide_idx + 1}"),
                        'chunk_index': slide_idx
                    },
                    start_char=0,
                    end_char=len(slide_content)
                )
                chunks.append(chunk)
            else:
                # Split large slide into multiple chunks
                slide_chunks = self._split_text(
                    slide_content,
                    metadata,
                    slide_meta.get('title', f"Slide {slide_idx + 1}"),
                    slide_idx
                )
                chunks.extend(slide_chunks)
        
        return chunks
    
    def _chunk_text(self, text: str, metadata: Dict) -> List[Chunk]:
        """Generic text chunking using recursive splitting."""
        return self._split_text(text, metadata, metadata.get('title', 'Document'))
    
    def _split_text(
        self,
        text: str,
        base_metadata: Dict,
        section_title: str,
        base_index: int = 0
    ) -> List[Chunk]:
        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            base_metadata: Base metadata for chunks
            section_title: Title of the section
            base_index: Base index for chunk numbering
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Try to split using separators
        for separator in self.separators:
            if separator == "":
                # Last resort: character-level splitting
                chunks = self._split_by_characters(text, base_metadata, section_title, base_index)
                break
            
            splits = text.split(separator)
            
            if len(splits) > 1:
                # Successfully split
                current_chunk = ""
                chunk_idx = 0
                
                for split in splits:
                    # Add separator back (except for first split)
                    if current_chunk:
                        current_chunk += separator
                    
                    # Check if adding this split would exceed chunk size
                    if len(current_chunk) + len(split) > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk = Chunk(
                            text=current_chunk.strip(),
                            chunk_id=f"{base_metadata['filename']}_chunk_{base_index}_{chunk_idx}",
                            metadata={
                                **base_metadata,
                                'section_title': section_title,
                                'chunk_index': chunk_idx
                            },
                            start_char=0,
                            end_char=len(current_chunk)
                        )
                        chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        if self.chunk_overlap > 0 and len(current_chunk) >= self.chunk_overlap:
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            current_chunk = overlap_text + separator + split
                        else:
                            current_chunk = split
                        
                        chunk_idx += 1
                    else:
                        current_chunk += split
                
                # Add remaining chunk
                if current_chunk.strip():
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"{base_metadata['filename']}_chunk_{base_index}_{chunk_idx}",
                        metadata={
                            **base_metadata,
                            'section_title': section_title,
                            'chunk_index': chunk_idx
                        },
                        start_char=0,
                        end_char=len(current_chunk)
                    )
                    chunks.append(chunk)
                
                break
        
        return chunks if chunks else [Chunk(
            text=text[:self.chunk_size].strip(),
            chunk_id=f"{base_metadata['filename']}_chunk_{base_index}_0",
            metadata={**base_metadata, 'section_title': section_title, 'chunk_index': 0},
            start_char=0,
            end_char=min(len(text), self.chunk_size)
        )]
    
    def _split_by_characters(
        self,
        text: str,
        base_metadata: Dict,
        section_title: str,
        base_index: int
    ) -> List[Chunk]:
        """Split text by characters as last resort."""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunk = Chunk(
                text=chunk_text.strip(),
                chunk_id=f"{base_metadata['filename']}_chunk_{base_index}_{chunk_idx}",
                metadata={
                    **base_metadata,
                    'section_title': section_title,
                    'chunk_index': chunk_idx
                },
                start_char=start,
                end_char=end
            )
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
            chunk_idx += 1
        
        return chunks

