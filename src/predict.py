import sys
import joblib
import os
import numpy as np

model_path = os.path.join("src/model", "rating_model.pkl")
model = joblib.load(model_path)


def predict(text: str) -> float:
    """Predict IMDb rating for a given text review."""
    predicted_rating = model.predict([text])[0]
    return float(np.clip(predicted_rating, 1.0, 10.0))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "This movie was amazing!"')
        sys.exit(1)

    review_text = sys.argv[1]
    predicted_rating = predict(review_text)
    print(
        f"Predicted IMDb rating for the review: {"Positive" if predicted_rating > 5 else "Negative"} {predicted_rating:.1f}"
    )
