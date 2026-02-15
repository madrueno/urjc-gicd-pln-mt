"""Advanced tokenizer using spaCy."""

import re

import spacy
from tqdm import tqdm


class SpacyTokenizer:
    """Advanced tokenizer using spaCy.

    Provides configurable preprocessing options including:
    - URL, email, and number removal
    - Lemmatization
    - Stopword removal
    - POS tag filtering

    Parameters
    ----------
    spacy_model : str, default='en_core_web_sm'
        spaCy model to use for NLP preprocessing.

    remove_noise : bool, default=True
        If True, removes URLs, email addresses, and numbers before tokenization.

    lemmatize : bool, default=True
        If True, returns lemmatized form of tokens (base form).

    remove_stopwords : bool, default=True
        If True, filters out stopwords using spaCy's built-in detection.

    pos_filter : list of str or None, default=['N', 'V', 'J', 'R']
        POS tag prefixes to keep (e.g., 'N' for nouns, 'V' for verbs).
        If None, no POS filtering is applied.
    """

    def __init__(
        self,
        spacy_model: str = 'en_core_web_sm',
        remove_noise: bool = True,
        lemmatize: bool = True,
        remove_stopwords: bool = True,
        pos_filter: list[str] | None = ['N', 'V', 'J', 'R'],
    ) -> None:
        """Initialize spaCy tokenizer.

        Parameters
        ----------
        spacy_model : str
            spaCy model name to load.
        remove_noise : bool
            If True, removes URLs, emails, and numbers.
        lemmatize : bool
            If True, returns lemmatized tokens.
        remove_stopwords : bool
            If True, filters out stopwords.
        pos_filter : list of str or None
            POS tag prefixes to keep. If None, no POS filtering.
        """
        self.nlp = spacy.load(spacy_model)
        self.remove_noise = remove_noise
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.pos_filter = pos_filter

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with configurable preprocessing.

        Applies preprocessing based on initialization parameters:

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        list[str]
            Processed tokens based on configuration.
        """
        # Remove noise if enabled
        if self.remove_noise:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            # Remove numbers
            text = re.sub(r'\d+', '', text)

        # Process with spaCy
        doc = self.nlp(text)

        # Extract tokens with filtering
        tokens = []
        for token in doc:
            # Skip punctuation and spaces (always)
            if token.is_punct or token.is_space:
                continue

            # Skip stopwords if enabled
            if self.remove_stopwords and token.is_stop:
                continue

            # POS filtering if enabled
            if self.pos_filter is not None and not any(token.tag_.startswith(prefix) for prefix in self.pos_filter):
                continue

            # Get token text (lemmatized or original)
            token_text = token.lemma_.lower() if self.lemmatize else token.text.lower()

            # Ensure non-empty
            if token_text:
                tokens.append(token_text)

        return tokens

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
