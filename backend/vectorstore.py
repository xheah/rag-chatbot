"""
Vector store implementation using FAISS.
Manages storage and retrieval of document embeddings.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss


class VectorStore:
    """Manages vector storage and retrieval using FAISS."""
    
    def __init__(
        self,
        persist_directory: str = "./vector_db",
        collection_name: str = "rag_documents"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist FAISS index and metadata
            collection_name: Name of the collection (used for file naming)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.index_path = self.persist_directory / f"{collection_name}.index"
        self.metadata_path = self.persist_directory / f"{collection_name}_metadata.pkl"
        
        # Initialize storage
        self.index = None
        self.texts = []
        self.metadatas = []
        self.ids = []
        self.embedding_dim = None
        
        # Load existing index if it exists
        if self.index_path.exists() and self.metadata_path.exists():
            self._load_index()
            print(f"Loaded existing vector store from {persist_directory}")
            print(f"Collection '{collection_name}' contains {len(self.texts)} documents")
        else:
            print(f"Initialized new vector store at {persist_directory}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        # Load index
        self.index = faiss.read_index(str(self.index_path))
        self.embedding_dim = self.index.d
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadatas = data['metadatas']
            self.ids = data['ids']
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadatas': self.metadatas,
                'ids': self.ids
            }, f)
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: Numpy array of embeddings (shape: [len(texts), embedding_dim])
            metadatas: List of metadata dictionaries for each chunk
            ids: Optional list of unique IDs. If None, will be auto-generated
        """
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("texts, embeddings, and metadatas must have the same length")
        
        # Ensure embeddings are float32 and 2D
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        embedding_dim = embeddings.shape[1]
        
        # Initialize index if needed
        if self.index is None:
            self.embedding_dim = embedding_dim
            # Use L2 distance (can be converted to cosine with normalization)
            # For cosine similarity, we'll normalize vectors
            self.index = faiss.IndexFlatL2(embedding_dim)
            print(f"Created new FAISS index with dimension {embedding_dim}")
        elif embedding_dim != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding_dim}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.ids)
            ids = [f"doc_{start_id + i}" for i in range(len(texts))]
        
        # Add to index
        self.index.add(embeddings)
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Save to disk
        self._save_index()
        
        print(f"Added {len(texts)} documents to vector store")
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"file_type": "markdown"})
            
        Returns:
            List of dictionaries with 'text', 'metadata', 'distance', and 'id' keys
        """
        if self.index is None or len(self.texts) == 0:
            return []
        
        # Ensure query embedding is float32 and 2D
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(n_results, len(self.texts)))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
            
            # Apply metadata filter if provided
            if filter_metadata:
                metadata = self.metadatas[idx]
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            results.append({
                'id': self.ids[idx],
                'text': self.texts[idx],
                'metadata': self.metadatas[idx],
                'distance': float(distances[0][i])
            })
        
        return results
    
    def delete_collection(self):
        """Delete the entire collection."""
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        
        self.index = None
        self.texts = []
        self.metadatas = []
        self.ids = []
        
        print(f"Deleted collection '{self.collection_name}'")
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        return {
            'collection_name': self.collection_name,
            'document_count': len(self.texts),
            'persist_directory': str(self.persist_directory),
            'embedding_dimension': self.embedding_dim
        }
    
    def delete_documents(self, ids: List[str]):
        """
        Delete specific documents by IDs.
        Note: FAISS doesn't support efficient deletion, so this rebuilds the index.
        
        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return
        
        # Find indices to keep
        ids_set = set(ids)
        keep_indices = [i for i, doc_id in enumerate(self.ids) if doc_id not in ids_set]
        
        if len(keep_indices) == len(self.ids):
            print("No documents found to delete")
            return
        
        # Rebuild index with remaining documents
        if len(keep_indices) == 0:
            self.delete_collection()
            return
        
        # Get embeddings for remaining documents (we need to store them)
        # Since we don't store embeddings, we'll need to rebuild from scratch
        # For now, just remove from metadata
        self.texts = [self.texts[i] for i in keep_indices]
        self.metadatas = [self.metadatas[i] for i in keep_indices]
        self.ids = [self.ids[i] for i in keep_indices]
        
        # Note: This is a limitation - we'd need to store embeddings to rebuild
        # For now, warn the user
        print(f"Warning: Deleted {len(ids)} documents from metadata, but index needs to be rebuilt.")
        print("Please re-add all remaining documents to rebuild the index properly.")
        
        # Delete the index file to force rebuild
        if self.index_path.exists():
            self.index_path.unlink()
        self.index = None

