import uvicorn
from fastapi import FastAPI, Query
from predict import predict

app = FastAPI(title="IMDb Rating Predictor")


@app.get("/predict")
def predict_rating(text: str = Query(..., description="Text of the review")):
    """Predict the IMDb rating from a review."""
    rating = predict(text)
    return {"review": text, "predicted_rating": round(rating, 1)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
