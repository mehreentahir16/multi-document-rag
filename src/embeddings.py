from openai import OpenAI
from typing import List
import time
from tqdm import tqdm
from .config import Config


class EmbeddingGenerator:
    """Generate embeddings using GitHub Models (OpenAI-compatible)"""
    
    def __init__(self):
        """Initialize OpenAI client for GitHub Models"""
        Config.validate()
        
        self.client = OpenAI(
            base_url=Config.GITHUB_API_BASE,
            api_key=Config.GITHUB_TOKEN,
        )
        
        self.model = Config.EMBEDDING_MODEL
        self.dimension = Config.PINECONE_DIMENSION
        
        print(f"🤖 Initialized embedding generator")
        print(f"   Model: {self.model}")
        print(f"   Dimension: {self.dimension}")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 50,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per API call (max 2048 for OpenAI)
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        print(f"\n🔄 Generating embeddings for {len(texts)} texts...")
        
        all_embeddings = []
        
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process with progress bar
        iterator = enumerate(batches, 1)
        if show_progress:
            iterator = tqdm(iterator, total=len(batches), desc="Embedding batches", unit="batch")
        
        for batch_num, batch in iterator:
            try:
                # Call GitHub Models API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting - be nice to free tier
                if batch_num < len(batches):  # Don't sleep after last batch
                    time.sleep(0.5)  # Small delay between batches
                
            except Exception as e:
                print(f"\nError in batch {batch_num}: {e}")
                raise
        
        print(f"Generated {len(all_embeddings)} embeddings")
        
        # Verify dimensions
        if all_embeddings and len(all_embeddings[0]) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch! "
                f"Expected {self.dimension}, got {len(all_embeddings[0])}"
            )
        
        return all_embeddings
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text (used for queries)"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f" Error generating embedding: {e}")
            raise