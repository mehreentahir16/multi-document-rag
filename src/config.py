import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration for RAG system"""
    
    # Directory Structure 
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    SRC_DIR = BASE_DIR / "src"
    
    # API Keys 
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    
    # GitHub Models Configuration 
    GITHUB_API_BASE: str = "https://models.inference.ai.azure.com"
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o").strip()
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "multi-document-rag")
    PINECONE_DIMENSION: int = 1536  # text-embedding-3-small
    PINECONE_METRIC: str = "cosine"
    
    # Document Processing
    # General chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Document-specific settings
    PDF_CHUNK_SIZE: int = 1800  # Technical papers need more context
    DOCX_CHUNK_SIZE: int = 1500  # Legal docs - standard
    EXCEL_CHUNK_SIZE: int = 500   # Tabular data - smaller chunks
    
    # RAG Configuration 
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    SIMILARITY_THRESHOLD: float = 0.5
    
    # Document Paths
    @classmethod
    def get_document_paths(cls) -> dict:
        """Get all challenge document paths"""
        return {
            "eu_ai_act": cls.DATA_DIR / "EU AI Act Doc.docx",
            "attention": cls.DATA_DIR / "Attention_is_all_you_need.pdf",
            "deepseek": cls.DATA_DIR / "Deepseek-r1.pdf",
            "inflation": cls.DATA_DIR / "Inflation Calculator.xlsx"
        }
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration and requirements"""
        errors = []
        
        # Check API keys
        if not cls.GITHUB_TOKEN:
            errors.append("GITHUB_TOKEN not set in .env")
        
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY not set in .env")
        
        # Check data directory
        if not cls.DATA_DIR.exists():
            errors.append(f"Data directory not found: {cls.DATA_DIR}")
        
        # Check documents
        docs = cls.get_document_paths()
        missing_docs = [name for name, path in docs.items() if not path.exists()]
        if missing_docs:
            errors.append(f"Missing documents: {', '.join(missing_docs)}")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f" {e}" for e in errors))
    
    @classmethod
    def summary(cls) -> str:
        """Get configuration summary"""
        docs = cls.get_document_paths()
        doc_status = "\n".join(
            f"  {'✅' if p.exists() else '❌'} {name}: {p.name}"
            for name, p in docs.items()
        )
        
        return f"""
Multi Document RAG Configuration
{'='*50}
Model: {cls.MODEL_NAME}
Embedding: {cls.EMBEDDING_MODEL}
Temperature: {cls.TEMPERATURE}

Pinecone:
  Index: {cls.PINECONE_INDEX_NAME}
  Dimension: {cls.PINECONE_DIMENSION}
  Metric: {cls.PINECONE_METRIC}

Chunking:
  PDF (Technical): {cls.PDF_CHUNK_SIZE} chars
  DOCX (Legal): {cls.DOCX_CHUNK_SIZE} chars
  Excel (Tabular): {cls.EXCEL_CHUNK_SIZE} chars
  Overlap: {cls.CHUNK_OVERLAP} chars

Documents:
{doc_status}

Retrieval:
  Top K: {cls.TOP_K}
  Threshold: {cls.SIMILARITY_THRESHOLD}
{'='*50}
"""