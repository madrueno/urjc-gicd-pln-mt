"""Simple tokenizer with optional noise removal."""

import re

from tqdm import tqdm


class SimpleTokenizer:
    """Simple tokenizer with optional noise removal.

    Provides lightweight tokenization with:
    - Lowercase conversion
    - Whitespace splitting
    - Optional noise removal (URLs, emails, numbers)

    Parameters
    ----------
    remove_noise : bool, default=True
        If True, removes URLs, email addresses, and numbers before tokenization.
    """

    def __init__(self, remove_noise: bool = True) -> None:
        """Initialize SimpleTokenizer.

        Parameters
        ----------
        remove_noise : bool
            If True, removes URLs, emails, and numbers before tokenization.
        """
        self.remove_noise_enabled = remove_noise

    @staticmethod
    def remove_noise(text: str) -> str:
        """Remove noise such as URLs, email addresses, and numbers from text.

        Parameters
        ----------
        text : str
            Input text to clean.

        Returns
        -------
        str
            Cleaned text with noise removed.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove numbers
        return re.sub(r'\d+', '', text)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with simple preprocessing.

        Applies preprocessing based on initialization parameters:
        - Noise removal (URLs, emails, numbers) if remove_noise=True
        - Lowercase conversion (always)
        - Whitespace splitting (always)

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        list[str]
            Lowercase tokens split on whitespace.
        """
        if self.remove_noise_enabled:
            text = self.remove_noise(text)

        return text.lower().split()

    def process(self, documents: list[str]) -> list[list[str]]:
        """Process multiple documents with progress tracking.

        Parameters
        ----------
        documents : list of str
            List of text documents to process.

        Returns
        -------
        list of list[str]
            List of tokenized documents.
        """
        results = []
        for doc in tqdm(documents, desc='Tokenizing', unit='doc'):
            results.append(self.tokenize(doc))
        return results
