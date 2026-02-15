"""BERT fine-tuning for text classification."""

import tempfile

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from spam_ham_detector.config import CUSTOM_MODELS_DIR


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Compute F1 score for evaluation.

    Parameters
    ----------
    eval_pred : EvalPrediction
        Evaluation predictions.

    Returns
    -------
    dict[str, float]
        Dictionary with F1 score.
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return {'f1': f1_score(labels, preds)}


class BERTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for BERT tokenized data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BERTClassifier:
    """
    Fine-tune BERT models for text classification.

    This class handles fine-tuning pre-trained BERT models for binary classification
    tasks with automatic model saving to avoid redundant training.

    Parameters
    ----------
    model_name : str, default="distilbert-base-uncased"
        Name of the pre-trained BERT model from HuggingFace.
    num_labels : int, default=2
        Number of classification labels.
    """

    def __init__(self, model_name: str = 'distilbert-base-uncased', num_labels: int = 2) -> None:
        self.model_name = model_name
        self.num_labels = num_labels

        self.model_folder = CUSTOM_MODELS_DIR / f'{model_name}-spam-ham'
        self.model_cache = self.model_folder / 'model'
        self.tokenizer_cache = self.model_folder / 'tokenizer'

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load a saved model from disk."""
        print(f'Loading saved model from {self.model_folder}...')

        if not self.model_cache.exists():
            raise FileNotFoundError(f'Model not found at {self.model_cache}. Train the model first.')

        if not self.tokenizer_cache.exists():
            raise FileNotFoundError(f'Tokenizer not found at {self.tokenizer_cache}. Train the model first.')

        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_cache), local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_cache), local_files_only=True)
        print('Model loaded successfully.')

    def train(
        self,
        X_train: list[str],
        y_train: list[int],
        X_dev: list[str],
        y_dev: list[int],
        num_epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 128,
    ) -> None:
        """
        Train a new BERT model and save it.

        Parameters
        ----------
        X_train : list of str
            Training texts.
        y_train : list of int
            Training labels.
        X_dev : list of str
            Development set texts.
        y_dev : list of int
            Development set labels.
        num_epochs : int, default=3
            Number of training epochs.
        batch_size : int, default=16
            Training batch size.
        max_length : int, default=128
            Maximum sequence length for tokenization.
        """
        print(f'Initializing {self.model_name} for fine-tuning...')

        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        # Tokenize datasets
        print('Tokenizing datasets...')
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
        dev_encodings = self.tokenizer(X_dev, truncation=True, padding=True, max_length=max_length)

        # Create PyTorch datasets
        train_dataset = BERTDataset(train_encodings, y_train)
        dev_dataset = BERTDataset(dev_encodings, y_dev)

        # Use temporary directory for training artifacts (auto-cleanup)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Training arguments
            training_args = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_ratio=0.06,
                eval_strategy='epoch',
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                greater_is_better=True,
                data_seed=42,
                seed=42,
            )

            # Create Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                compute_metrics=compute_metrics,
            )
            # Fine-tune the model
            print(f'Fine-tuning {self.model_name}...')
            trainer.train()

        # Save model
        print(f'Saving model to {self.model_folder}...')
        self.model_folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(self.model_cache))
        self.tokenizer.save_pretrained(str(self.tokenizer_cache))
        print('Model saved successfully.')

    def predict_proba(self, texts: list[str], max_length: int = 128):
        """
        Predict class probabilities for input texts.

        Parameters
        ----------
        texts : list of str
            Texts to classify.
        max_length : int, default=128
            Maximum sequence length for tokenization.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError('Model not loaded. Call load() or train() first.')

        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        device = next(self.model.parameters()).device
        encodings = {k: v.to(device) for k, v in encodings.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return probabilities.cpu().numpy()

    def predict(self, texts: list[str], max_length: int = 128):
        """
        Predict labels for input texts.

        Parameters
        ----------
        texts : list of str
            Texts to classify.
        max_length : int, default=128
            Maximum sequence length for tokenization.

        Returns
        -------
        predictions : list of int
            Predicted class labels.
        """
        probabilities = self.predict_proba(texts, max_length=max_length)
        return probabilities.argmax(axis=-1).tolist()
