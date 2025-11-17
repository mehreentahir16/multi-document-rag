from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from .config import Config
from .embeddings import EmbeddingGenerator
from .vector_store import PineconeVectorStore


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with metadata"""
    text: str
    score: float
    source: str
    doc_type: str
    metadata: Dict[str, Any]
    
    def get_citation(self) -> str:
        """Generate citation string for this chunk"""
        if self.doc_type == 'pdf':
            page = self.metadata.get('page', 'unknown')
            return f"{self.source} (page {page})"
        elif self.doc_type == 'docx':
            section = self.metadata.get('section', 'unknown')
            return f"{self.source} (section {section})"
        elif self.doc_type == 'excel':
            sheet = self.metadata.get('sheet', 'unknown')
            row = self.metadata.get('row', 'unknown')
            return f"{self.source} (sheet: {sheet}, row: {row})"
        return self.source


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources"""
    answer: str
    sources: List[RetrievedChunk]
    query: str
    
    def get_formatted_sources(self) -> str:
        """Get formatted source citations"""
        if not self.sources:
            return "No sources found."
        
        citations = []
        for i, chunk in enumerate(self.sources, 1):
            citations.append(f"{i}. {chunk.get_citation()} (relevance: {chunk.score:.3f})")
        
        return "\n".join(citations)


class RAGSystem:
    """Complete RAG system: Retrieval + Generation"""
    
    def __init__(self):
        """Initialize RAG components"""
        Config.validate()
        
        # Initialize components
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = PineconeVectorStore()
        
        # Initialize LLM for generation
        self.llm_client = OpenAI(
            base_url=Config.GITHUB_API_BASE,
            api_key=Config.GITHUB_TOKEN,
        )
        
        self.model = Config.MODEL_NAME
        self.temperature = Config.TEMPERATURE
        
        print(" RAG system initialized")
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            filter_dict: Optional metadata filter
        
        Returns:
            List of retrieved chunks with scores
        """
        if top_k is None:
            top_k = Config.TOP_K
        
        # Generate query embedding
        query_embedding = self.embedding_gen.generate_single_embedding(query)
        
        # Query Pinecone
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for match in results:
            retrieved_chunks.append(RetrievedChunk(
                text=match.metadata.get('text', ''),
                score=match.score,
                source=match.metadata.get('source', 'unknown'),
                doc_type=match.metadata.get('doc_type', 'unknown'),
                metadata=match.metadata
            ))
        
        return retrieved_chunks
    
    def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[RetrievedChunk]
    ) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User question
            retrieved_chunks: Retrieved document chunks
        
        Returns:
            Generated answer
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            citation = chunk.get_citation()
            context_parts.append(f"[Source {i}: {citation}]\n{chunk.text}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on provided document context.

                        IMPORTANT RULES:
                        1. Only use information from the provided context
                        2. If the answer isn't in the context, clearly state that
                        3. Always cite your sources using [Source X] format
                        4. Be concise but comprehensive
                        5. If multiple sources support your answer, mention all relevant ones
                        """
        
        user_prompt = f"""Context from documents:

{context}

Question: {query}

Please answer the question based on the context above. Cite sources using [Source X] format."""

        try:
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=Config.MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Complete RAG query: Retrieve + Generate
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            filter_dict: Optional metadata filter
        
        Returns:
            RAGResponse with answer and sources
        """
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, top_k, filter_dict)
        
        # Step 2: Generate answer
        answer = self.generate_answer(query, retrieved_chunks)
        
        # Return complete response
        return RAGResponse(
            answer=answer,
            sources=retrieved_chunks,
            query=query
        )
