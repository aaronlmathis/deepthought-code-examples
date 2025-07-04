from sklearn.metrics import classification_report


def evaluate_model(model, X_test, y_test, log_results=True):
    """
    Evaluates a trained classification model on a test dataset and logs the results.

    This function generates predictions using the provided model, computes a variety of
    classification metrics, and optionally logs a detailed classification report.
    It is designed to provide both a summary of key metrics for benchmarking and
    a full report for in-depth analysis.

    Args:
        model: Trained scikit-learn compatible classifier with a .predict() method.
        X_test: Features of the test set (array-like or DataFrame).
        y_test: True labels for the test set (array-like or Series).
        log_results (bool): If True, logs the weighted recall and full classification report.

    Returns:
        dict: A dictionary containing accuracy, macro-averaged, and weighted-averaged
              precision, recall, and F1-score metrics.
    """
    # Generate predictions for the test set using the trained model
    y_pred = model.predict(X_test)

    # Compute the classification report as a dictionary for easy metric extraction
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']  # Overall accuracy of the model
    weighted_recall = report['weighted avg']['recall']  # Recall averaged by support (number of true instances for each label)

    if log_results:
        # Log the weighted recall for quick reference
        print(f"Test Set Weighted Recall: {weighted_recall:.4f}")
        # Log the full, human-readable classification report for detailed analysis
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Collect key metrics for benchmarking and further analysis
    metrics = {
        'accuracy': accuracy,
        'precision_macro': report['macro avg']['precision'],      # Precision averaged equally across classes
        'recall_macro': report['macro avg']['recall'],            # Recall averaged equally across classes
        'f1_score_macro': report['macro avg']['f1-score'],        # F1-score averaged equally across classes
        'precision_weighted': report['weighted avg']['precision'],# Precision averaged by class support
        'recall_weighted': weighted_recall,                       # Recall averaged by class support
        'f1_score_weighted': report['weighted avg']['f1-score'],  # F1-score averaged by class support
    }
    return metrics
