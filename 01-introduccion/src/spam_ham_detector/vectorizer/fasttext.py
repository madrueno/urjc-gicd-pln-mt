"""FastText embedding extraction for text data."""

import numpy as np
from gensim.models.fasttext import load_facebook_model


class FastText:
    """Extract sentence embeddings using pre-trained FastText word vectors.

    Uses mean pooling to aggregate word vectors into document vectors.
    FastText can generate vectors for out-of-vocabulary words using subword info.

    Parameters
    ----------
    model_path : str, default="models/external/wiki.simple.bin"
        Path to the FastText .bin model file.
    """

    def __init__(self, model_path: str = 'models/external/wiki.simple.bin') -> None:
        print(f'Loading FastText model: {model_path}')
        self.model = load_facebook_model(model_path).wv

    def embed(self, tokens: list[str]) -> np.ndarray:
        """Convert a single tokenized document to its embedding vector.

        Uses mean pooling: computes the average of all word vectors.
        FastText generates vectors for OOV words using subword information.

        Parameters
        ----------
        tokens : list of str
            List of tokens (pre-tokenized text).

        Returns
        -------
        np.ndarray
            Embedding vector of shape (embedding_dim,).
        """
        if not tokens:
            return np.zeros(self.model.vector_size)

        # Get word vectors for all tokens
        vectors = [self.model[token] for token in tokens]

        # Mean pooling: average all word vectors
        return np.mean(vectors, axis=0)

    def embed_documents(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        """Convert multiple tokenized documents to embedding vectors.

        Parameters
        ----------
        tokenized_docs : list of list of str
            List of tokenized documents (each document is a list of tokens).

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_documents, embedding_dim).
        """
        embeddings = []
        for tokens in tokenized_docs:
            embedding = self.embed(tokens)
            embeddings.append(embedding)
        return np.array(embeddings)
