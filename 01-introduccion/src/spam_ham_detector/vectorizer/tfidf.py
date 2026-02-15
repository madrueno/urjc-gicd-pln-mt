"""TF-IDF vectorization for pre-tokenized text data."""

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:
    """TF-IDF vectorizer for pre-tokenized documents.

    Wrapper around sklearn's TfidfVectorizer that works with pre-tokenized input.
    Converts tokenized documents to TF-IDF sparse matrices.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to sklearn's TfidfVectorizer.
    """

    def __init__(self, **kwargs) -> None:
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, **kwargs)

    def fit_transform(self, tokenized_docs: list[list[str]]) -> sp.csr_matrix:
        """Fit vectorizer and transform tokenized documents to TF-IDF matrix.

        Parameters
        ----------
        tokenized_docs : list of list of str
            List of tokenized documents (each document is a list of tokens).

        Returns
        -------
        scipy.sparse matrix
            TF-IDF matrix of shape (n_documents, n_features).
        """
        return self.vectorizer.fit_transform(tokenized_docs)

    def transform(self, tokenized_docs: list[list[str]]) -> sp.csr_matrix:
        """Transform tokenized documents to TF-IDF matrix.

        Parameters
        ----------
        tokenized_docs : list of list of str
            List of tokenized documents (each document is a list of tokens).

        Returns
        -------
        scipy.sparse matrix
            TF-IDF matrix of shape (n_documents, n_features).
        """
        return self.vectorizer.transform(tokenized_docs)
