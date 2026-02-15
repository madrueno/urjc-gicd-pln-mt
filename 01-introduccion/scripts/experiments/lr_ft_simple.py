"""Training pipeline for FastText-based spam classification with simple preprocessing.

This experiment uses simple preprocessing (lowercase + whitespace split + noise removal)
to establish a baseline for comparing preprocessing strategies with FastText embeddings.

WARNING: Test set results should ONLY be examined after selecting hyperparameters
based on dev set performance. Using test results for model selection leads to
overfitting and invalidates your evaluation.
"""

from sklearn.linear_model import LogisticRegression

from spam_ham_detector import (
    CommentsDataset,
    FastText,
    SimpleTokenizer,
    evaluate_metrics,
    print_experiment_summary,
    save_predictions,
)


def main():
    """Load dataset, extract embeddings with simple preprocessing, and evaluate."""
    # ===== DATASET / PIPELINE =====
    dataset = CommentsDataset()

    tokenizer = SimpleTokenizer(remove_noise=True)
    embedder = FastText()

    # ========== TRAIN SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TRAIN SET' + '\n' + '=' * 70)
    X_train_tokens = tokenizer.process(dataset.X_train)
    X_train_emb = embedder.embed_documents(X_train_tokens)

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_emb, dataset.y_train)

    y_train_pred = classifier.predict(X_train_emb)
    y_train_proba = classifier.predict_proba(X_train_emb)[:, 1]
    train_metrics = evaluate_metrics(y_train_pred, dataset.y_train)

    # ========== DEV SET ==========
    print('=' * 70 + '\n' + 'PROCESSING DEV SET (for model selection)' + '\n' + '=' * 70)
    X_dev_tokens = tokenizer.process(dataset.X_dev)
    X_dev_emb = embedder.embed_documents(X_dev_tokens)

    y_dev_pred = classifier.predict(X_dev_emb)
    y_dev_proba = classifier.predict_proba(X_dev_emb)[:, 1]
    dev_metrics = evaluate_metrics(y_dev_pred, dataset.y_dev)

    # ========== TEST SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TEST SET (for final reporting)' + '\n' + '=' * 70)
    X_test_tokens = tokenizer.process(dataset.X_test)
    X_test_emb = embedder.embed_documents(X_test_tokens)

    y_test_pred = classifier.predict(X_test_emb)
    y_test_proba = classifier.predict_proba(X_test_emb)[:, 1]
    test_metrics = evaluate_metrics(y_test_pred, dataset.y_test)

    # ========== FINAL SUMMARY ==========
    experiment = 'lr-ft-simple'
    save_predictions(dataset.X_train, dataset.y_train, y_train_pred, y_train_proba, experiment, 'train')
    save_predictions(dataset.X_dev, dataset.y_dev, y_dev_pred, y_dev_proba, experiment, 'dev')
    save_predictions(dataset.X_test, dataset.y_test, y_test_pred, y_test_proba, experiment, 'test')
    print_experiment_summary(train_metrics, dev_metrics, test_metrics, experiment_name=experiment)


if __name__ == '__main__':
    main()
