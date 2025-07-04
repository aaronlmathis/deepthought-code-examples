from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump
import os



def train_model(X_train, y_train, preprocessor, save_path='models/voting_model.joblib'):
    """
    Builds and trains a full pipeline with a VotingClassifier using default parameters.

    This function constructs a machine learning pipeline that integrates data preprocessing
    and model training in a single workflow. The core classifier is a VotingClassifier,
    which combines predictions from multiple base estimators to improve robustness and accuracy.
    The trained pipeline is saved to disk for later inference or evaluation.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        y_train (pd.Series or np.ndarray): Training target vector.
        preprocessor (ColumnTransformer): Preprocessing pipeline for feature transformation.
        save_path (str): File path where the trained model pipeline will be saved.

    Returns:
        model (Pipeline): The trained scikit-learn pipeline, ready for prediction.
    """
    print("--- Training a robust Voting Classifier with default parameters ---")

    # Define the base estimators for the ensemble.
    # Logistic Regression: Linear model, good baseline for binary classification.
    # K-Nearest Neighbors: Non-parametric, captures local structure in data.
    # Support Vector Classifier: Effective in high-dimensional spaces, uses probability estimates.
    estimators = [
        ('lr', LogisticRegression(max_iter=10000, random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # Create a soft voting classifier, which averages predicted probabilities.
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')

    # Build a pipeline that first preprocesses the data, then fits the ensemble classifier.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])

    # Ensure the directory for saving the model exists.
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Fit the pipeline on the training data.
    pipeline.fit(X_train, y_train)
    model = pipeline

    # Persist the trained pipeline to disk using joblib for efficient serialization.
    dump(model, save_path)
    print(f"Model saved to {save_path}")
    return model
