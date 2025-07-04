import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Generates and saves a confusion matrix heatmap.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels from the model.
        model_name (str): Name of the model for labeling the plot.
        save_path (str, optional): If provided, saves the plot to this path; otherwise, displays it.

    This function visualizes the confusion matrix, which summarizes the performance of a classification
    algorithm by showing the counts of true positives, true negatives, false positives, and false negatives.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if save_path:
        # Ensure the directory exists before saving the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_roc_curve(model, X_test, y_test, model_name, save_path=None):
    """
    Generates and saves a Receiver Operating Characteristic (ROC) curve.

    Args:
        model: Trained classifier with a predict_proba method.
        X_test: Test features.
        y_test: True labels for the test set.
        model_name (str): Name of the model for labeling the plot.
        save_path (str, optional): If provided, saves the plot to this path; otherwise, displays it.

    The ROC curve illustrates the diagnostic ability of a binary classifier system as its discrimination
    threshold is varied. The Area Under the Curve (AUC) provides a single measure of overall performance.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")

    if save_path:
        # Ensure the directory exists before saving the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved ROC curve to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(model, X_test, y_test, model_name, save_path=None):
    """
    Generates and saves a Precision-Recall curve. This is particularly useful for
    imbalanced datasets.

    Args:
        model: Trained classifier with a predict_proba method.
        X_test: Test features.
        y_test: True labels for the test set.
        model_name (str): Name of the model for labeling the plot.
        save_path (str, optional): If provided, saves the plot to this path; otherwise, displays it.

    The Precision-Recall curve is especially informative when dealing with imbalanced classes,
    as it focuses on the performance with respect to the positive class.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)

    if save_path:
        # Ensure the directory exists before saving the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved Precision-Recall curve to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, preprocessor, model_name, save_path=None):
    """
    Extracts and plots feature importances from the Logistic Regression
    component of the VotingClassifier. This helps with model interpretability.

    Args:
        model: Trained scikit-learn pipeline containing a VotingClassifier.
        preprocessor: The preprocessing pipeline used for feature transformation.
        model_name (str): Name of the model for labeling the plot.
        save_path (str, optional): If provided, saves the plot to this path; otherwise, displays it.

    This function is designed to work with a pipeline whose final estimator is a VotingClassifier
    containing a Logistic Regression estimator named 'lr'. It visualizes the top 20 features
    by absolute coefficient magnitude, aiding in understanding which features most influence predictions.
    """
    try:
        # Get feature names from the preprocessor pipeline.
        feature_names = preprocessor.get_feature_names_out()

        # Extract the logistic regression model and its coefficients.
        # The model is a pipeline; the classifier is the second step.
        voting_clf = model.named_steps['classifier']
        lr_model = voting_clf.named_estimators_['lr']
        importances = lr_model.coef_[0]

        # Create a DataFrame for easier plotting, showing top 20 features by absolute importance.
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', key=abs, ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title(f'Top 20 Feature Importances for {model_name} (from Logistic Regression)')
        plt.xlabel('Coefficient (Importance)')
        plt.ylabel('Feature')
        plt.tight_layout()

        if save_path:
            # Ensure the directory exists before saving the plot
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved feature importance plot to {save_path}")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        # Log errors for debugging and provide a hint about possible causes.
        log.error(f"Could not generate feature importance plot: {e}")
        log.error("This is likely because the model is not the expected VotingClassifier or the 'lr' estimator is missing.")
