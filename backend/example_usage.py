"""
Example usage of the RAG pipeline.
Shows how to use the components individually and together.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from document_loader import DocumentLoader
from chunker import TextChunker
from embedder import Embedder
from vectorstore import VectorStore
from rag_engine import RAGEngine


def example_full_pipeline():
    """Example of using the complete RAG pipeline."""
    
    # 1. Load documents
    loader = DocumentLoader()
    documents = loader.load_directory("../data")
    
    # 2. Chunk documents
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    # 3. Generate embeddings
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_batch(texts)
    
    # 4. Store in vector database
    vector_store = VectorStore(persist_directory="./vector_db")
    metadatas = [chunk.metadata for chunk in all_chunks]
    ids = [chunk.chunk_id for chunk in all_chunks]
    
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    # 5. Create RAG engine
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedder=embedder,
        llm_provider="openai",
        model_name="gpt-3.5-turbo"
    )
    
    # 6. Ask questions
    query = "What is a data structure?"
    response = rag_engine.generate_response(query, n_results=5)
    
    print(f"Question: {query}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources: {', '.join(response['sources'])}")


def example_individual_components():
    """Example of using components individually."""
    
    # Load a single document
    loader = DocumentLoader()
    doc = loader.load_document("../data/Introduction to DSA (W1L1 21 10 25) 2935d306702780bc8463db33de944a7f.md")
    print(f"Loaded: {doc['metadata']['title']}")
    print(f"Content length: {len(doc['content'])} characters")
    
    # Chunk the document
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings for chunks
    embedder = Embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_batch(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Store in vector database
    vector_store = VectorStore()
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]
    
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    # Search
    query = "What is data?"
    query_embedding = embedder.embed_text(query)
    results = vector_store.search(query_embedding, n_results=3)
    
    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text'][:100]}...")


if __name__ == "__main__":
    print("Choose an example:")
    print("1. Full pipeline with RAG")
    print("2. Individual components")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        example_full_pipeline()
    elif choice == "2":
        example_individual_components()
    else:
        print("Invalid choice")

