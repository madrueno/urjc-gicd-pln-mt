"""SBERT embedding extraction for text data."""

import numpy as np
from sentence_transformers import SentenceTransformer


class SBERT:
    """
    Extract sentence embeddings using SBERT models.

    SBERT produces semantically meaningful sentence embeddings
    using Siamese BERT-based networks. Unlike word embeddings that require
    pooling, SBERT directly generates sentence-level representations.

    Parameters
    ----------
    model_name : str, default="all-mpnet-base-v2"
        Name of the pre-trained SBERT model from HuggingFace.
    """

    def __init__(self, model_name: str = 'all-mpnet-base-v2') -> None:
        print(f'Loading SBERT model: {model_name}')
        self.model = SentenceTransformer(model_name)
        print('Model loaded successfully.')

    def embed(self, text: str) -> np.ndarray:
        """
        Convert a single text document to its embedding vector.

        Parameters
        ----------
        text : str
            Raw text document (no tokenization needed).

        Returns
        -------
        np.ndarray
            Embedding vector of shape (embedding_dim,).
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Convert multiple text documents to embedding vectors.

        Parameters
        ----------
        texts : list of str
            List of raw text documents (no tokenization needed).

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_documents, embedding_dim).
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
