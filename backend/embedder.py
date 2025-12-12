"""
Text embedding generation using sentence transformers.
Converts text chunks into vector embeddings for semantic search.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generates embeddings for text chunks."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
                       Options: "all-MiniLM-L6-v2" (fast, 384 dims),
                               "all-mpnet-base-v2" (better quality, 768 dims)
            device: Device to run model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Model loaded successfully on {device}")
    
    def embed_text(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings (single or list)
        """
        if isinstance(text, str):
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        else:
            embeddings = self.model.encode(text, convert_to_numpy=True, show_progress_bar=True)
            return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            Numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Get dimension by encoding a dummy text
        dummy_embedding = self.embed_text("test")
        return len(dummy_embedding)

