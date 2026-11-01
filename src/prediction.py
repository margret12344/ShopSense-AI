import joblib
from pathlib import Path

from src.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_ecommerce_classifier.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.joblib"


class ProductClassifier:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

        self.metadata = {}

        if METADATA_PATH.exists():
            self.metadata = joblib.load(METADATA_PATH)

        self.categories = self.metadata.get(
            "categories",
            list(getattr(self.model, "classes_", []))
        )

    def predict(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter a valid product description.")

        cleaned_text = clean_text(text)

        if not cleaned_text:
            raise ValueError("No meaningful text remains after preprocessing.")

        prediction = self.model.predict([cleaned_text])[0]

        result = {
            "prediction": prediction,
            "cleaned_text": cleaned_text,
            "scores": self._get_scores(cleaned_text)
        }

        return result

    def _get_scores(self, cleaned_text):

        try:
            if hasattr(self.model, "predict_proba"):

                probabilities = self.model.predict_proba(
                    [cleaned_text]
                )[0]

                return {
                    category: float(score)
                    for category, score in zip(
                        self.model.classes_,
                        probabilities
                    )
                }

            if hasattr(self.model, "decision_function"):

                scores = self.model.decision_function(
                    [cleaned_text]
                )[0]

                if len(self.model.classes_) == 2:
                    scores = [-scores, scores]

                return {
                    category: float(score)
                    for category, score in zip(
                        self.model.classes_,
                        scores
                    )
                }

        except Exception:
            return {}

        return {}

    def predict_batch(self, texts):

        cleaned_texts = [
            clean_text(text)
            for text in texts
        ]

        predictions = self.model.predict(cleaned_texts)

        return predictions