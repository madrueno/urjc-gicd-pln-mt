"""Dataset loader for YouTube comment spam classification."""

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_ham_detector.config import PROCESSED_DATA_DIR


class CommentsDataset:
    """
    Loader for YouTube comment spam classification datasets.

    Attributes
    ----------
    X_train : list[str]
        Training features (comment text)
    y_train : list[int]
        Training labels (0=ham, 1=spam)
    X_dev : list[str]
        Dev/validation features
    y_dev : list[int]
        Dev/validation labels
    X_test : list[str]
        Test features (comment text)
    y_test : list[int]
        Test labels (0=ham, 1=spam)
    """

    def __init__(self):
        """Load train/test/dev splits automatically."""
        # Load datasets from processed data directory
        train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
        test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')

        # Test set
        self.X_test = test_df['CONTENT'].to_list()
        self.y_test = test_df['CLASS'].to_list()

        # Split train into train/dev (dev same size as test)
        X_train_full = train_df['CONTENT'].to_list()
        y_train_full = train_df['CLASS'].to_list()

        dev_size = len(self.X_test) / len(X_train_full)

        self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(
            X_train_full, y_train_full, test_size=dev_size, random_state=42, stratify=y_train_full
        )
