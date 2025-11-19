import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

from .config import Config
from .chunking import ProcessedChunk

class PineconeVectorStore:
    """Manage Pinecone vector store operations"""
    
    def __init__(self):
        """Initialize Pinecone and connect to index"""
        Config.validate()
        
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.index = None
        
        self._connect_to_index()
    
    def _connect_to_index(self):
        """Connect to existing index or create if doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name in existing_indexes:
            print(f"Connected to index: {self.index_name}")
            self.index = self.pc.Index(self.index_name)
            
            # Show current stats
            stats = self.get_stats()
            print(f"Current vectors: {stats['total_vectors']}")
        else:
            print(f"Creating index: {self.index_name}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=Config.PINECONE_DIMENSION,
                metric=Config.PINECONE_METRIC,
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            
            # Wait for index readiness
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            print("Index created successfully!")
    
    def upload_chunks(
        self,
        chunks: List[ProcessedChunk],
        embeddings: List[List[float]],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Upload chunks with their embeddings to Pinecone
        
        Args:
            chunks: List of ProcessedChunk objects
            embeddings: List of embedding vectors (must match chunks length)
            batch_size: Vectors per batch (max 100 for Starter)
            show_progress: Show progress bar
        
        Returns:
            Upload statistics
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch!"
            )
        
        # reset index contents before uploading (we want to avoid duplication in case of a re-run)
        print(f"\n Resetting Pinecone index before upload...")
        self.clear_index()
        
        print(f"\n Uploading {len(chunks)} vectors to Pinecone...")
        
        uploaded = 0
        total = len(chunks)
        
        # Create batches
        batches = []
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batches.append((i, batch_end))
        
        # Upload with progress
        iterator = batches
        if show_progress:
            iterator = tqdm(batches, desc="Uploading batches", unit="batch")
        
        for start, end in iterator:
            # Prepare batch data
            batch_data = []
            for j in range(start, end):
                chunk = chunks[j]
                embedding = embeddings[j]
                
                batch_data.append({
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.text  # Store text in metadata for retrieval
                    }
                })
            
            # Upload batch
            try:
                self.index.upsert(vectors=batch_data)
                uploaded += len(batch_data)
            except Exception as e:
                print(f"\n Error uploading batch: {e}")
                raise
        
        print(f"Successfully uploaded {uploaded} vectors")
        
        # Wait a moment for indexing
        time.sleep(2)
        
        # Verify upload
        stats = self.get_stats()
        print(f"Total vectors in index: {stats['total_vectors']}")
        
        return {
            "uploaded": uploaded,
            "total_in_index": stats['total_vectors'],
            "index_name": self.index_name
        }
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query Pinecone for similar vectors"""
        if top_k is None:
            top_k = Config.TOP_K
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            return results.matches
            
        except Exception as e:
            print(f"Query error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }
    
    def clear_index(self) -> bool:
        """Clear all vectors (use with caution!)"""
        try:
            self.index.delete(delete_all=True)
            print(f"Cleared all vectors from {self.index_name}")
            return True
        except Exception as e:
            print(f"Error clearing index: {e}")
            return False