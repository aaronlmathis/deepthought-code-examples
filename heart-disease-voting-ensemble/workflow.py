import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from preprocess import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model
from visualize import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_feature_importance

import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output/logs

class PredictionWorkflow:
    """
    A class to manage the end-to-end prediction workflow, including
    data loading, preprocessing, model training, evaluation, and reporting.

    This class encapsulates all major steps required for a reproducible machine learning
    pipeline in a modular and maintainable way. It is designed to be the main orchestrator
    for the heart disease prediction project, ensuring that each stage is executed in the
    correct order and that results are logged and visualized for analysis.
    """

    def __init__(self, model_path='models/voting_classifier_model.joblib',
                confusion_matrix_path='reports/confusion_matrix.png',
                roc_curve_path='reports/roc_curve.png',
                pr_curve_path='reports/precision_recall_curve.png',
                feature_importance_path='reports/feature_importance.png'):
        """
        Initializes the PredictionWorkflow with file paths for saving the model
        and reports.

        Args:
            model_path (str): Path to save the trained model.
            confusion_matrix_path (str): Path to save the confusion matrix plot.
            roc_curve_path (str): Path to save the ROC curve plot.
            pr_curve_path (str): Path to save the Precision-Recall curve plot.
            feature_importance_path (str): Path to save the feature importance plot.
        """
        self.model_path = model_path
        self.confusion_matrix_path = confusion_matrix_path
        self.roc_curve_path = roc_curve_path
        self.pr_curve_path = pr_curve_path
        self.feature_importance_path = feature_importance_path
        self.model = None
        self.preprocessor = None

    def run(self):
        """
        Executes the complete prediction workflow.

        This method is the main entry point for running the workflow. It sequentially
        calls all major steps: data loading/preprocessing, model training, evaluation,
        and reporting. Each step is logged for traceability and debugging.
        """
        print("Starting the prediction workflow...")
        self.load_and_preprocess()
        self.train()
        self.evaluate()
        self.report()
        print("Prediction workflow completed.")

    def load_and_preprocess(self):
        """
        Loads and preprocesses the data, splitting it into training and testing sets.

        This method loads the raw heart disease datasets, applies feature engineering,
        and constructs a preprocessing pipeline. It then splits the data into training
        and test sets using stratified sampling to preserve class distribution.
        """
        print("Loading and preprocessing data...")
        X_engineered, y_engineered, self.preprocessor = load_and_preprocess_data()
        self.X_full = X_engineered
        self.y_full = y_engineered

        # Stratified split ensures that both train and test sets have similar class distributions.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42, stratify=self.y_full
        )
        print("Data loading and preprocessing complete.")
        print("-" * 50)

    def train(self):
        """
        Trains the model using the training data.

        This method builds a machine learning pipeline that includes preprocessing and
        a VotingClassifier ensemble. The trained pipeline is saved to disk for future use.
        """
        print("Training the model...")
        self.model = train_model(self.X_train, self.y_train, self.preprocessor, save_path=self.model_path)
        print("Model training complete.")

    def evaluate(self):
        """
        Evaluates the trained model on the test data and performs cross-validation.

        This method computes a variety of classification metrics on the held-out test set
        and also performs cross-validation on the full dataset to estimate generalization
        performance. Results are stored for reporting.
        """
        print("Evaluating the model...")
        self.evaluation_metrics = evaluate_model(self.model, self.X_test, self.y_test)

        # Perform 5-fold stratified cross-validation using weighted recall as the scoring metric.
        self.cross_validation_scores = cross_val_score(
            self.model, self.X_full, self.y_full, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='recall_weighted'
        )
        print("Model evaluation complete.")
        print("-" * 50)

    def report(self):
        """
        Generates and saves the confusion matrix, ROC curve, Precision-Recall curve,
        and feature importance plots. Logs evaluation metrics and cross-validation results.

        This method provides visual and quantitative feedback on model performance,
        supporting both interpretability and reproducibility for the project.
        """
        print("Generating and saving reports...")

        # Generate predictions for the test set for confusion matrix visualization.
        y_pred = self.model.predict(self.X_test)

        # Plot and save confusion matrix.
        plot_confusion_matrix(self.y_test, y_pred,
                              model_name='VotingClassifier',
                              save_path=self.confusion_matrix_path)
        # Plot and save ROC curve.
        plot_roc_curve(self.model, self.X_test, self.y_test,
                       model_name='VotingClassifier', save_path=self.roc_curve_path)
        # Plot and save Precision-Recall curve.
        plot_precision_recall_curve(self.model, self.X_test, self.y_test,
                                    model_name='VotingClassifier', save_path=self.pr_curve_path)
        # Plot and save feature importance (from Logistic Regression component).
        plot_feature_importance(self.model, self.model.named_steps['preprocessor'],
                                model_name='VotingClassifier', save_path=self.feature_importance_path)

        # Log summary metrics and cross-validation results for transparency and reproducibility.
        print("*" * 50)
        print("*    Heart Disease Prediction Workflow Report    *")
        print("*" * 50)
        print("Evaluation Metrics:")
        for metric, value in self.evaluation_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("-" * 50)
        print(f"Cross-validation recall: {self.cross_validation_scores.mean():.4f} ± {self.cross_validation_scores.std():.4f}")
        print("Reporting complete.")

if __name__ == '__main__':
    workflow = PredictionWorkflow()
    workflow.run()