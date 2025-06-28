# 🎬 IMDb Rating Predictor

This project uses a simple machine learning pipeline to **predict IMDb ratings** from movie reviews using **TF-IDF** and **Ridge Regression**.

---

## 📦 Installation

Make sure you have **Python 3.10+** and a virtual environment activated.

```bash
python -m venv <evn-name>

# for windows
.\<evn-name>\Scripts\activate

# for linux
source ./<evn-name>/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Train the Model

First, extract the training data files, **data.zip**, inside src folder. Then:

```bash
python src/train.py

# Output:
# 📊 MSE on test set: <number>
# ✅ Model trained and saved to 'model/rating_model.pkl'
```

Then, to run the model as on console, run:

```bash
python src/predict.py "string"

# Example Usage:
# $ pyton src/predict.py "I like this movie"
# Predicted IMDb rating for the review: Positive 9.8
```

If you want to use as an API, run:

```bash
python src/run.py
# This will launch the server at:
# 📍 http://0.0.0.0:8000

```

You can request to `https://localhost:8000/predict?text="I like this movie"`, then the api will return:

```json
{
  "review": "This movie was incredible!",
  "predicted_rating": 8.9
}
```

## 🛠 Project Structure

```txt
mini-ai-project/
│
├── src/
│ ├── data/ # Contains raw IMDb review data
│ │ └── imdb/
│ │ │ └── train/
│ │ │ │ └── pos/ # 5000+ positive review .txt files
│ │ │ │ └── neg/ # 5000+ negative review .txt files
│ │ │ └── test/
│ │ │ │ ├── pos/ # Test set positive reviews
│ │ │ │ └── neg/ # Test set negative reviews
│ │
│ ├── model/
│ │ └── rating_model.pkl # Trained machine learning model
│ │
│ ├── train.py # Trains and saves the regression model
│ ├── predict.py # Loads model and predicts rating from input
│ ├── utils.py # Text loading, preprocessing, and helpers
│ └── run.py # FastAPI app entry point for predictions
│
├── setup.py # CLI entry: train or run the app
├── requirements.txt # Project dependencies
└── README.md # Main project documentation

```
