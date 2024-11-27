import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from dotenv import load_dotenv
import utils.logging as logger

load_dotenv()

def fetch_data_from_excel(file_path):
    """
    Fetches data from an Excel file.
    The Excel file should have columns: 'phrase' and 'label'.
    """
    try:
        logger.log_message("Reading data from the Excel file", level="info")
        df = pd.read_excel(file_path)

        # Ensure the required columns exist
        required_columns = ["phrase", "label"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"The Excel file must contain the following columns: {required_columns}")

        logger.log_message("Data read successfully from the Excel file", level="info")
        return df
    except Exception as e:
        logger.log_message(f"Error reading data from the Excel file: {str(e)}", level="error")
        raise

def train_l1_model(file_path):
    try:
        logger.log_message("Starting L1 model training process", level="info")

        df = fetch_data_from_excel(file_path)

        X = df["phrase"]
        y = df["label"]

        class_weights = compute_class_weight("balanced", classes=y.unique(), y=y)
        class_weight_dict = dict(zip(y.unique(), class_weights))
        logger.log_message(f"Computed class weights: {class_weight_dict}", level="info")

        vectorizer = TfidfVectorizer()
        X_vect = vectorizer.fit_transform(X)

        scaler = MaxAbsScaler()
        X_vect_scaled = scaler.fit_transform(X_vect)

        X_train, X_test, y_train, y_test = train_test_split(X_vect_scaled, y, test_size=0.2, stratify=y, random_state=42)
        model = LogisticRegression(max_iter=2000, class_weight=class_weight_dict)
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        logger.log_message(f"Best model parameters: {grid_search.best_params_}", level="info")
        logger.log_message(f"Model evaluation:\n{classification_report(y_test, y_pred)}", level="info")

        model_filename = os.getenv("L1_MODEL_SAVE_PATH", "./models/L1/logistic_regression_model.pkl")
        vectorizer_filename = os.getenv("L1_VECTORIZER_FILE_PATH_SAVE", "./models/L1/tfidf_vectorizer.pkl")
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_filename), exist_ok=True)

        with open(model_filename, "wb") as model_file:
            pickle.dump(best_model, model_file)
        with open(vectorizer_filename, "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

        logger.log_message(f"Model saved to {model_filename}", level="info")
        logger.log_message(f"Vectorizer saved to {vectorizer_filename}", level="info")
    except Exception as e:
        logger.log_message(f"Error during L1 model training: {str(e)}", level="error")
        raise