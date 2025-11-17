from data_loading_and_chunking_pipeline import DocumentPipeline
from .embeddings import EmbeddingGenerator
from .vector_store import PineconeVectorStore
from typing import Any, Dict, Tuple, List
from .chunking import ProcessedChunk


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
        print(" STARTING FULL INDEXING PIPELINE")
        print("="*70)
        
        # Step 1: Load and chunk documents
        print("\n STEP 1: Document Processing")
        print("-"*70)
        chunks = self.doc_pipeline.process_documents()
        
        if save_chunks:
            self.doc_pipeline.save_chunks(chunks, "processed_chunks.json")
        
        # Step 2: Generate embeddings
        print("\n STEP 2: Embedding Generation")
        print("-"*70)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_gen.generate_embeddings(texts)
        
        # Step 3: Upload to Pinecone
        print("\n STEP 3: Pinecone Upload")
        print("-"*70)
        upload_stats = self.vector_store.upload_chunks(chunks, embeddings)
        
        # Summary
        print("\n" + "="*70)
        print(" INDEXING PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n Summary:")
        print(f"  Documents Processed: 4 (PDF x2, DOCX x1, Excel x1)")
        print(f"  Total Chunks Created: {len(chunks)}")
        print(f"  Embeddings Generated: {len(embeddings)}")
        print(f"  Vectors Uploaded: {upload_stats['uploaded']}")
        print(f"  Pinecone Index: {upload_stats['index_name']}")
        print(f"  Total Vectors in Index: {upload_stats['total_in_index']}")
        
        return chunks, upload_stats

def build_rag_index():
    """
    Main script to build the complete RAG index
    Run this once to populate Pinecone with all document embeddings
    """
    
    print("\n" + "Multi Document RAG - INDEX BUILDER" + "\n")
    
    try:
        # Run complete pipeline
        pipeline = IndexingPipeline()
        chunks, stats = pipeline.run_full_pipeline()
        
        print("\n" + "="*70)
        print(" SUCCESS! Your RAG system is ready!")
        print("="*70)
        print(f"\n Chunk data saved to: processed_chunks.json")
        print(f" Pinecone index: {stats['index_name']}")
        print(f" Total searchable chunks: {stats['total_in_index']}")
        
        return True
        
    except Exception as e:
        print(f"\n PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n Troubleshooting:")
        print("  1. Check your .env file has GITHUB_TOKEN and PINECONE_API_KEY")
        print("  2. Verify documents are in data/ folder")
        print("  3. Ensure Pinecone index exists (multi-document-rag)")
        print("  4. Check GitHub Models rate limits")
        
        return False


if __name__ == "__main__":
    success = build_rag_index()