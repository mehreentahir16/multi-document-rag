import re
import hashlib
from .config import Config
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class ProcessedChunk:
    """Final processed chunk ready for embedding"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'metadata': self.metadata
        }


class DocumentChunker:
    """Smart document chunking with document-type awareness"""
    
    def __init__(self):
        """Initialize chunkers for different document types"""
        
        # Technical papers (PDFs) - need more context
        self.pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.PDF_CHUNK_SIZE,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\nFigure", "\nReferences", "\n",". ", " ", ""],
            keep_separator=True
        )
        
        # Legal documents (DOCX) - standard chunking
        self.docx_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.DOCX_CHUNK_SIZE,
            chunk_overlap=200,
            length_function=len,
            separators=["\n", ". ", " "],
            keep_separator=True
        )
        
        # Tabular data (Excel) - smaller chunks
        self.excel_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.EXCEL_CHUNK_SIZE,
            chunk_overlap=50,  # Less overlap for structured data
            length_function=len,
            separators=["\n", ", ", " "],
            keep_separator=True
        )
    
    def _generate_chunk_id(self, text: str, source: str, index: int) -> str:
        """Generate unique, deterministic chunk ID"""
        # Create hash from text content for uniqueness
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        # Clean source name
        source_clean = re.sub(r'[^\w\-]', '_', source.replace('.', '_'))
        return f"{source_clean}_{index}_{text_hash}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _chunk_pdf(self, doc_chunks: List[Dict[str, Any]], doc_name: str) -> List[ProcessedChunk]:
        """Chunk technical papers (PDFs) with page-aware splitting"""
        print(f" Chunking PDF: {doc_name}")
        
        processed_chunks = []
        chunk_index = 0
        
        for doc_chunk in doc_chunks:
            text = self._clean_text(doc_chunk['text'])
            page = doc_chunk.get('page', 'unknown')
            source = doc_chunk['source']
            
            # Split the page text into chunks
            splits = self.pdf_splitter.split_text(text)
            
            for split_text in splits:
                if len(split_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                chunk_id = self._generate_chunk_id(split_text, source, chunk_index)
                
                processed_chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    text=split_text,
                    metadata={
                        'source': source,
                        'doc_name': doc_name,
                        'doc_type': 'pdf',
                        'page': page,
                        'chunk_index': chunk_index,
                        'char_count': len(split_text)
                    }
                ))
                chunk_index += 1
        
        print(f" Created {len(processed_chunks)} chunks")
        return processed_chunks
    
    def _chunk_docx(self, doc_chunks: List[Dict[str, Any]], doc_name: str) -> List[ProcessedChunk]:
        """Chunk legal documents (DOCX) with section-aware splitting"""
        print(f"  📄 Chunking DOCX: {doc_name}")
        
        processed_chunks = []
        chunk_index = 0
        
        for doc_chunk in doc_chunks:
            text = self._clean_text(doc_chunk['text'])
            section = doc_chunk.get('section', 'unknown')
            source = doc_chunk['source']
            
            # Split the section text into chunks
            splits = self.docx_splitter.split_text(text)
            
            for split_text in splits:
                if len(split_text.strip()) < 50:
                    continue
                
                chunk_id = self._generate_chunk_id(split_text, source, chunk_index)
                
                processed_chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    text=split_text,
                    metadata={
                        'source': source,
                        'doc_name': doc_name,
                        'doc_type': 'docx',
                        'section': section,
                        'chunk_index': chunk_index,
                        'char_count': len(split_text)
                    }
                ))
                chunk_index += 1
        
        print(f"    ✅ Created {len(processed_chunks)} chunks")
        return processed_chunks
    
    def _chunk_excel(self, doc_chunks: List[Dict[str, Any]], doc_name: str) -> List[ProcessedChunk]:
        """Process tabular data (Excel) - minimal chunking"""
        print(f"  📊 Processing Excel: {doc_name}")
        
        processed_chunks = []
        
        for chunk_index, doc_chunk in enumerate(doc_chunks):
            text = self._clean_text(doc_chunk['text'])
            
            # For Excel, we typically keep rows together
            # Only split if a single row description is too long
            if len(text) > Config.EXCEL_CHUNK_SIZE:
                splits = self.excel_splitter.split_text(text)
            else:
                splits = [text]
            
            for split_text in splits:
                if len(split_text.strip()) < 20:
                    continue
                
                chunk_id = self._generate_chunk_id(
                    split_text, 
                    doc_chunk['source'], 
                    chunk_index
                )
                
                processed_chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    text=split_text,
                    metadata={
                        'source': doc_chunk['source'],
                        'doc_name': doc_name,
                        'doc_type': 'excel',
                        'sheet': doc_chunk.get('sheet', 'unknown'),
                        'row': doc_chunk.get('row', 'unknown'),
                        'chunk_index': chunk_index,
                        'char_count': len(split_text)
                    }
                ))
        
        print(f"    ✅ Created {len(processed_chunks)} chunks")
        return processed_chunks
    
    def process_all_documents(
        self, 
        loaded_docs: Dict[str, List[Dict[str, Any]]]
    ) -> List[ProcessedChunk]:
        """
        Process all loaded documents into optimally chunked pieces
        
        Args:
            loaded_docs: Output from DocumentLoader.load_all_documents()
        
        Returns:
            List of ProcessedChunk objects ready for embedding
        """
        print("\n" + "="*60)
        print("✂️  CHUNKING ALL DOCUMENTS")
        print("="*60 + "\n")
        
        all_chunks = []
        
        # Process each document with appropriate strategy
        for doc_name, doc_chunks in loaded_docs.items():
            if not doc_chunks:
                print(f"  ⚠️  Skipping {doc_name} (no content)")
                continue
            
            # Determine document type and use appropriate chunker
            doc_type = doc_chunks[0].get('doc_type', '')
            
            if doc_type == 'pdf':
                chunks = self._chunk_pdf(doc_chunks, doc_name)
            elif doc_type == 'docx':
                chunks = self._chunk_docx(doc_chunks, doc_name)
            elif doc_type == 'excel':
                chunks = self._chunk_excel(doc_chunks, doc_name)
            else:
                print(f"  ⚠️  Unknown doc type for {doc_name}: {doc_type}")
                continue
            
            all_chunks.extend(chunks)
        
        print("\n" + "="*60)
        print(f"✅ CHUNKING COMPLETE: {len(all_chunks)} total chunks ready")
        print("="*60 + "\n")
        
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Get statistics about chunked documents"""
        if not chunks:
            return {}
        
        # Group by document
        by_doc = {}
        for chunk in chunks:
            doc_name = chunk.metadata['doc_name']
            if doc_name not in by_doc:
                by_doc[doc_name] = []
            by_doc[doc_name].append(chunk)
        
        # Calculate statistics
        stats = {
            'total_chunks': len(chunks),
            'by_document': {},
            'avg_chunk_size': sum(c.metadata['char_count'] for c in chunks) / len(chunks),
            'min_chunk_size': min(c.metadata['char_count'] for c in chunks),
            'max_chunk_size': max(c.metadata['char_count'] for c in chunks),
        }
        
        for doc_name, doc_chunks in by_doc.items():
            stats['by_document'][doc_name] = {
                'count': len(doc_chunks),
                'doc_type': doc_chunks[0].metadata['doc_type'],
                'avg_size': sum(c.metadata['char_count'] for c in doc_chunks) / len(doc_chunks)
            }
        
        return stats
