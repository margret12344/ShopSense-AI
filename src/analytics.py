import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


def load_csv(filename):

    file_path = REPORTS_DIR / filename

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


def load_model_metadata():

    file_path = MODELS_DIR / "model_metadata.joblib"

    if not file_path.exists():
        return {}

    return joblib.load(file_path)


def load_model_comparison():

    return load_csv("model_comparison_results.csv")


def load_cross_validation():

    return load_csv("cross_validation_results.csv")


def load_misclassified():

    return load_csv("misclassified_examples.csv")


def get_project_statistics():

    metadata = load_model_metadata()
    comparison = load_model_comparison()

    return {
        "selected_model": metadata.get(
            "selected_model",
            "Not available"
        ),
        "categories": metadata.get("categories", []),
        "comparison": comparison,
        "number_of_models": len(comparison)
    }