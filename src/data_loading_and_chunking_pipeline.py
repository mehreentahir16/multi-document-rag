import json
from typing import List, Dict, Any

from .document_loader import DocumentLoader
from .chunking import DocumentChunker, ProcessedChunk

class DocumentPipeline:
    """Complete document processing pipeline"""
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
    
    def process_documents(self) -> List[ProcessedChunk]:
        """
        Complete pipeline: Load → Chunk → Return processed chunks
        
        Returns:
            List of ProcessedChunk objects ready for embedding
        """
        #Load all documents
        loaded_docs = self.loader.load_all_documents()
        
        #Chunk documents
        chunks = self.chunker.process_all_documents(loaded_docs)
        
        #Show statistics
        stats = self.chunker.get_chunking_stats(chunks)
        self._print_stats(stats)
        
        return chunks
    
    def _print_stats(self, stats: Dict[str, Any]):
        """Print processing statistics"""
        print("\nPROCESSING STATISTICS")
        print("="*60)
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Average Chunk Size: {stats['avg_chunk_size']:.0f} chars")
        print(f"Size Range: {stats['min_chunk_size']} - {stats['max_chunk_size']} chars")
        print("\nBy Document:")
        print("-"*60)
        
        for doc_name, doc_stats in stats['by_document'].items():
            print(f"{doc_name} ({doc_stats['doc_type']}):")
            print(f"Chunks: {doc_stats['count']}")
            print(f"Avg Size: {doc_stats['avg_size']:.0f} chars")
        
        print("="*60)
    
    def save_chunks(self, chunks: List[ProcessedChunk], output_path: str = "processed_chunks.json"):
        """Save processed chunks to JSON for inspection"""
        data = [chunk.to_dict() for chunk in chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(chunks)} chunks to {output_path}")
