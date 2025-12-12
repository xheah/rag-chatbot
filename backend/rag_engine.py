"""
RAG Engine: Combines retrieval and generation for question answering.
"""

import os
from typing import List, Dict, Optional
import numpy as np
from embedder import Embedder
from vectorstore import VectorStore


class RAGEngine:
    """Main RAG engine that combines retrieval and generation."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        llm_provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG engine.
        
        Args:
            vector_store: Initialized VectorStore instance
            embedder: Initialized Embedder instance
            llm_provider: LLM provider ("openai", "anthropic", or "local")
            model_name: Model name to use
            api_key: API key for the LLM provider
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize LLM client based on provider
        self.llm_client = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "openai":
            try:
                import openai
                if self.api_key:
                    openai.api_key = self.api_key
                elif os.getenv("OPENAI_API_KEY"):
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                else:
                    print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                return openai
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                if self.api_key:
                    anthropic.api_key = self.api_key
                elif os.getenv("ANTHROPIC_API_KEY"):
                    anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
                return anthropic
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def retrieve(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            n_results: Number of results to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant document chunks
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def generate_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate a prompt for the LLM with context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata'].get('filename', 'Unknown')
            section = chunk['metadata'].get('section_title', '')
            text = chunk['text']
            
            context_parts.append(
                f"[Source {i}: {source}"
                + (f" - {section}" if section else "")
                + f"]\n{text}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}

Question: {query}

Instructions:
- Answer the question using only the information from the context documents above.
- If the context doesn't contain enough information to answer the question, say so.
- Cite which source(s) you used in your answer.
- Be concise but thorough.

Answer:"""
        
        return prompt
    
    def generate_response(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict:
        """
        Generate a response using RAG pipeline.
        
        Args:
            query: User query
            n_results: Number of context chunks to retrieve
            filter_metadata: Optional metadata filters
            temperature: LLM temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with 'answer', 'sources', and 'context_chunks' keys
        """
        # Retrieve relevant context
        context_chunks = self.retrieve(query, n_results=n_results, filter_metadata=filter_metadata)
        
        if not context_chunks:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'context_chunks': []
            }
        
        # Generate prompt
        prompt = self.generate_prompt(query, context_chunks)
        
        # Generate response using LLM
        answer = self._call_llm(prompt, temperature=temperature, max_tokens=max_tokens)
        
        # Extract sources
        sources = list(set([
            chunk['metadata'].get('filename', 'Unknown')
            for chunk in context_chunks
        ]))
        
        return {
            'answer': answer,
            'sources': sources,
            'context_chunks': context_chunks
        }
    
    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Call the LLM with the prompt."""
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error calling OpenAI API: {str(e)}"
        
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
                
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.content[0].text.strip()
            except Exception as e:
                return f"Error calling Anthropic API: {str(e)}"
        
        else:
            return "LLM provider not properly configured."

