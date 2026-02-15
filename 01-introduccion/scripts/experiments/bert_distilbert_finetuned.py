"""Training pipeline for BERT-based spam classification with fine-tuning.

This experiment fine-tunes a BERT model from scratch for spam classification.

Trained model is stored in models/custom directory.

WARNING: Test set results should ONLY be examined after selecting hyperparameters
based on dev set performance.
"""

import argparse

from spam_ham_detector import (
    BERTClassifier,
    CommentsDataset,
    evaluate_metrics,
    print_experiment_summary,
    save_predictions,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT model for spam classification')
    parser.add_argument(
        '--train', action='store_true', help='Train a new model (WARNING: will overwrite existing model if present)'
    )
    return parser.parse_args()


def main(args):
    """Load dataset, fine-tune BERT, and evaluate on all splits."""
    # ===== DATASET / PIPELINE =====
    dataset = CommentsDataset()

    bert = BERTClassifier(model_name='distilbert-base-uncased')

    if args.train:
        print('Training new model (will overwrite if exists)...')
        bert.train(dataset.X_train, dataset.y_train, dataset.X_dev, dataset.y_dev)
    else:
        print('Loading existing model...')
        try:
            bert.load()
        except FileNotFoundError:
            print('Model not found. Training new model...')
            bert.train(dataset.X_train, dataset.y_train, dataset.X_dev, dataset.y_dev)

    # ========== TRAIN SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TRAIN SET' + '\n' + '=' * 70)
    train_predictions = bert.predict(dataset.X_train)
    y_train_proba = bert.predict_proba(dataset.X_train)[:, 1]
    train_results = evaluate_metrics(train_predictions, dataset.y_train)

    # ========== DEV SET ==========
    print('=' * 70 + '\n' + 'PROCESSING DEV SET (for model selection)' + '\n' + '=' * 70)
    dev_predictions = bert.predict(dataset.X_dev)
    y_dev_proba = bert.predict_proba(dataset.X_dev)[:, 1]
    dev_results = evaluate_metrics(dev_predictions, dataset.y_dev)

    # ========== TEST SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TEST SET (for final reporting)' + '\n' + '=' * 70)
    test_predictions = bert.predict(dataset.X_test)
    y_test_proba = bert.predict_proba(dataset.X_test)[:, 1]
    test_results = evaluate_metrics(test_predictions, dataset.y_test)

    # ========== FINAL SUMMARY ==========
    experiment = 'bert-distilbert-finetuned'
    save_predictions(dataset.X_train, dataset.y_train, train_predictions, y_train_proba, experiment, 'train')
    save_predictions(dataset.X_dev, dataset.y_dev, dev_predictions, y_dev_proba, experiment, 'dev')
    save_predictions(dataset.X_test, dataset.y_test, test_predictions, y_test_proba, experiment, 'test')
    print_experiment_summary(train_results, dev_results, test_results, experiment_name=experiment)


if __name__ == '__main__':
    args = parse_args()
    main(args)
