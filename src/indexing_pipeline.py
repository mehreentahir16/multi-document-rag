from typing import Any, Dict, Tuple, List

from .chunking import ProcessedChunk
from .embeddings import EmbeddingGenerator
from .vector_store import PineconeVectorStore
from .data_loading_and_chunking_pipeline import DocumentPipeline


class IndexingPipeline:
    """Complete pipeline: Load → Chunk → Embed → Upload"""
    
    def __init__(self):
        self.doc_pipeline = DocumentPipeline()
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = PineconeVectorStore()
    
    def run_full_pipeline(self, save_chunks: bool = True) -> Tuple[List[ProcessedChunk], Dict[str, Any]]:
        """
        Run complete indexing pipeline
        
        Returns:
            Tuple of (chunks, upload_stats)
        """
        print("\n" + "="*70)
        print("STARTING FULL INDEXING PIPELINE...")
        print("="*70)
        
        #Load and chunk documents
        print("\n Processing Documents...")
        print("-"*70)
        chunks = self.doc_pipeline.process_documents()
        
        if save_chunks:
            self.doc_pipeline.save_chunks(chunks, "processed_chunks.json")
        
        #Generate embeddings
        print("\n Generating Embeddings...")
        print("-"*70)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_gen.generate_embeddings(texts)
        
        # Upload to Pinecone
        print("\n Uploading to Pinecone...")
        print("-"*70)
        upload_stats = self.vector_store.upload_chunks(chunks, embeddings)
        
        # Summary
        print("\n" + "="*70)
        print(" INDEXING PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n Summary:")
        print(f"Documents Processed: 4 (PDF x2, DOCX x1, Excel x1)")
        print(f"Total Chunks Created: {len(chunks)}")
        print(f"Embeddings Generated: {len(embeddings)}")
        print(f"Vectors Uploaded: {upload_stats['uploaded']}")
        print(f"Pinecone Index: {upload_stats['index_name']}")
        print(f"Total Vectors in Index: {upload_stats['total_in_index']}")
        
        return chunks, upload_stats
