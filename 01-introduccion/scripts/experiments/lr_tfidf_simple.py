"""Training pipeline for TF-IDF spam classification with simple preprocessing.

This experiment uses simple preprocessing (lowercase + whitespace split + noise removal)
to establish a baseline for comparing preprocessing strategies.

WARNING: Test set results should ONLY be examined after selecting hyperparameters
based on dev set performance. Using test results for model selection leads to
overfitting and invalidates your evaluation.
"""

from sklearn.linear_model import LogisticRegression

from spam_ham_detector import (
    TFIDF,
    CommentsDataset,
    SimpleTokenizer,
    evaluate_metrics,
    print_experiment_summary,
    save_predictions,
)


def main():
    """Load dataset, train TF-IDF classifier with simple preprocessing, and evaluate."""
    # ===== DATASET / PIPELINE =====
    dataset = CommentsDataset()

    tokenizer = SimpleTokenizer(remove_noise=True)
    vectorizer = TFIDF(max_features=5000)

    # ========== TRAIN SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TRAIN SET' + '\n' + '=' * 70)
    X_train_tokens = tokenizer.process(dataset.X_train)
    X_train_vec = vectorizer.fit_transform(X_train_tokens)

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_vec, dataset.y_train)

    y_train_pred = classifier.predict(X_train_vec)
    y_train_proba = classifier.predict_proba(X_train_vec)[:, 1]
    train_metrics = evaluate_metrics(y_train_pred, dataset.y_train)

    # ========== DEV SET ==========
    print('=' * 70 + '\n' + 'PROCESSING DEV SET (for model selection)' + '\n' + '=' * 70)
    X_dev_tokens = tokenizer.process(dataset.X_dev)
    X_dev_vec = vectorizer.transform(X_dev_tokens)

    y_dev_pred = classifier.predict(X_dev_vec)
    y_dev_proba = classifier.predict_proba(X_dev_vec)[:, 1]
    dev_metrics = evaluate_metrics(y_dev_pred, dataset.y_dev)

    # ========== TEST SET ==========
    print('=' * 70 + '\n' + 'PROCESSING TEST SET (for final reporting)' + '\n' + '=' * 70)
    X_test_tokens = tokenizer.process(dataset.X_test)
    X_test_vec = vectorizer.transform(X_test_tokens)

    y_test_pred = classifier.predict(X_test_vec)
    y_test_proba = classifier.predict_proba(X_test_vec)[:, 1]
    test_metrics = evaluate_metrics(y_test_pred, dataset.y_test)

    # ========== FINAL SUMMARY ==========
    experiment = 'lr-tfidf-simple'
    save_predictions(dataset.X_train, dataset.y_train, y_train_pred, y_train_proba, experiment, 'train')
    save_predictions(dataset.X_dev, dataset.y_dev, y_dev_pred, y_dev_proba, experiment, 'dev')
    save_predictions(dataset.X_test, dataset.y_test, y_test_pred, y_test_proba, experiment, 'test')
    print_experiment_summary(train_metrics, dev_metrics, test_metrics, experiment_name=experiment)


if __name__ == '__main__':
    main()
