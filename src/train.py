import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils import load_imdb_data, load_lexicon, AvgSentimentScore


def main():
    train_dir = os.path.join("src", "data", "imdb", "train")
    test_dir = os.path.join("src", "data", "imdb", "test")

    train_texts, train_ratings = load_imdb_data(train_dir)
    test_texts, test_ratings = load_imdb_data(test_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        train_texts, train_ratings, test_size=0.2, random_state=42
    )

    lexicon = load_lexicon("src/data/imdb/imdb.vocab", "src/data/imdb/imdbEr.txt")

    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("tfidf", TfidfVectorizer(max_features=10000)),
                        ("avg_sentiment", AvgSentimentScore(lexicon=lexicon)),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(test_texts)
    mse = mean_squared_error(test_ratings, predictions)
    print(f"📊 MSE on test set: {mse:.2f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/rating_model.pkl")
    print("✅ Model trained and saved to 'model/rating_model.pkl'")


if __name__ == "__main__":
    main()
