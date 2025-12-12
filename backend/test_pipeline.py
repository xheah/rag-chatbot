"""
Test script for the RAG pipeline.
Tests document loading, chunking, embedding, and retrieval.
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


def test_pipeline(data_directory: str = "../data"):
    """Test the complete RAG pipeline."""
    
    print("=" * 60)
    print("Testing RAG Pipeline")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n[1/5] Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_directory(data_directory)
    print(f"Loaded {len(documents)} documents")
    
    if not documents:
        print("No documents found! Please add documents to the data directory.")
        return
    
    # Step 2: Chunk documents
    print("\n[2/5] Chunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  - {doc['metadata']['filename']}: {len(chunks)} chunks")
    
    print(f"Total chunks: {len(all_chunks)}")
    
    # Step 3: Generate embeddings
    print("\n[3/5] Generating embeddings...")
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_batch(texts, batch_size=32)
    print(f"Generated embeddings: shape {embeddings.shape}")
    
    # Step 4: Store in vector database
    print("\n[4/5] Storing in vector database...")
    vector_store = VectorStore(persist_directory="./vector_db")
    
    # Prepare metadata and IDs
    metadatas = [chunk.metadata for chunk in all_chunks]
    ids = [chunk.chunk_id for chunk in all_chunks]
    
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Vector store contains {vector_store.get_collection_info()['document_count']} documents")
    
    # Step 5: Test retrieval
    print("\n[5/5] Testing retrieval...")
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedder=embedder,
        llm_provider="openai",  # Change to "anthropic" if preferred
        model_name="gpt-3.5-turbo"
    )
    
    # Test queries
    test_queries = [
        "What is a data structure?",
        "What are the differences between arrays and linked structures?",
        "Explain stacks and queues"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Retrieval (without LLM)")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag_engine.retrieve(query, n_results=3)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('filename', 'Unknown')
            distance = result.get('distance', 'N/A')
            preview = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
            print(f"  {i}. [{source}] (distance: {distance:.4f})")
            print(f"     {preview}")
    
    print("\n" + "=" * 60)
    print("Pipeline test completed successfully!")
    print("=" * 60)
    print("\nNote: To test full RAG with LLM generation, set OPENAI_API_KEY")
    print("      environment variable and use rag_engine.generate_response()")


if __name__ == "__main__":
    # Get data directory from command line or use default
    # Get the project root (parent of backend directory)
    project_root = Path(__file__).parent.parent
    default_data_dir = project_root / "data"
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(default_data_dir)
    test_pipeline(data_dir)

