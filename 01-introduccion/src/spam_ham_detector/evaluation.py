"""Evaluation metrics for classification models."""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from spam_ham_detector.config import RESULTS_DIR


def evaluate_metrics(y_pred, y_true):
    """
    Calculate and print classification metrics.

    Parameters
    ----------
    y_pred : array-like
        Predicted labels
    y_true : array-like
        True labels

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and f1 scores
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print()
    print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def print_experiment_summary(train_metrics, dev_metrics, test_metrics, experiment_name, metric_prefix=''):
    """
    Print final experiment summary with all results.

    Parameters
    ----------
    train_metrics : dict
        Training set metrics
    dev_metrics : dict
        Development set metrics
    test_metrics : dict
        Test set metrics
    experiment_name : str
        Name of the experiment for display
    metric_prefix : str
        Prefix for metric keys (e.g., 'eval_' for BERT models)
    """
    accuracy_key = f'{metric_prefix}accuracy'
    f1_key = f'{metric_prefix}f1'

    print('=' * 70)
    print(f'EXPERIMENT SUMMARY: {experiment_name}')
    print('=' * 70)
    print(f'Train - Accuracy: {train_metrics[accuracy_key]:.4f}, F1: {train_metrics[f1_key]:.4f}')
    print(f'Dev   - Accuracy: {dev_metrics[accuracy_key]:.4f}, F1: {dev_metrics[f1_key]:.4f}')
    print(f'Test  - Accuracy: {test_metrics[accuracy_key]:.4f}, F1: {test_metrics[f1_key]:.4f}')
    print('=' * 70)

    print('REMINDER: Use dev set results to select best model.')
    print('Only report test set results as final performance.')
    print('=' * 70)


def save_predictions(texts, y_true, y_pred, y_scores, experiment_name, split_name):
    """Save predictions to the results directory.

    Parameters
    ----------
    texts : array-like
        Input texts.
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_scores : array-like
        Prediction scores/probabilities.
    experiment_name : str
        Name of the experiment.
    split_name : str
        Split name (train/dev/test).

    """
    experiment_dir = RESULTS_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    output_path = experiment_dir / f'{split_name}.csv'

    df = pd.DataFrame({'text': list(texts), 'y_true': list(y_true), 'y_pred': list(y_pred), 'y_scores': list(y_scores)})
    df.to_csv(output_path, index=False)
    print(f'Saved predictions to {output_path}')
