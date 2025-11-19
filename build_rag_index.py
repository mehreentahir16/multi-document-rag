from src.indexing_pipeline import IndexingPipeline

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
        print(" SUCCESS! RAG system is ready!")
        print("="*70)
        print(f"\n Chunk data saved to: processed_chunks.json")
        print(f"Pinecone index: {stats['index_name']}")
        print(f"Total searchable chunks: {stats['total_in_index']}")
        
        return True
        
    except Exception as e:
        print(f"\n PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n Troubleshooting:")
        print("1. Check your .env file has GITHUB_TOKEN and PINECONE_API_KEY")
        print("2. Verify documents are in data/ folder")
        print("3. Ensure Pinecone index exists (multi-document-rag)")
        print("4. Check GitHub Models rate limits")
        
        return False


if __name__ == "__main__":
    success = build_rag_index()