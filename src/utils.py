import os
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
import numpy as np


def extract_rating_from_filename(filename):
    match = re.match(r"\d+_(\d+)\.txt", filename)
    return int(match.group(1)) if match else None


def load_imdb_data(data_dir):
    texts = []
    ratings = []

    for label_dir in ["pos", "neg"]:
        path = os.path.join(data_dir, label_dir)
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                rating = extract_rating_from_filename(filename)
                if rating is not None:
                    with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                        texts.append(f.read())
                        ratings.append(rating)
    return texts, ratings


def load_lexicon(path_vocab, path_ratings):
    with open(path_vocab, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    with open(path_ratings, "r", encoding="utf-8") as f:
        scores = [float(line.strip()) for line in f]

    return dict(zip(vocab, scores))


class AvgSentimentScore(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            words = word_tokenize(text.lower())
            ratings = [self.lexicon.get(w, 5.0) for w in words]  # Neutral default
            avg_score = np.mean(ratings) if ratings else 5.0
            features.append([avg_score])
        return np.array(features)
