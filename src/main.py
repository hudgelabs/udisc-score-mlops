import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Disc Golf Score Predictor")

# Initialize variables as None
reg = None
latest_rating = None
model = None

# Safety checks for GitHub Actions
if os.path.exists('data/course_registry.csv'):
    reg = pd.read_csv('data/course_registry.csv')

if os.path.exists('data/processed_scores.csv'):
    processed = pd.read_csv('data/processed_scores.csv')
    latest_rating = processed['PlayerRating_At_Time'].iloc[-1]

if os.path.exists('models/model.pkl'):
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)


class PredictRequest(BaseModel):
    CourseName: str
    LayoutName: str
    Month: int
    Duration_Min: float
    Wind: float
    Temp: float


@app.get("/")
def health_check():
    return {
        "status": "Online",
        "model_loaded": model is not None,
        "registry_loaded": reg is not None
    }


@app.post("/predict")
def predict(data: PredictRequest):
    if model is None or reg is None:
        raise HTTPException(status_code=503, detail="Model or Registry not loaded")

    course = reg[(reg.CourseName == data.CourseName) &
                 (reg.LayoutName == data.LayoutName)]

    if course.empty:
        raise HTTPException(status_code=404, detail="Course not found")

    feat_cols = [
        'PlayerRating_At_Time', 'ParRating', 'GlobalAvgScore',
        'CoursePar', 'Month', 'Duration_Min', 'Wind', 'Temp'
    ]

    input_data = [[
        latest_rating,
        course['ParRating'].iloc[0],
        course['GlobalAvgScore'].iloc[0],
        course['CoursePar'].iloc[0],
        data.Month, data.Duration_Min, data.Wind, data.Temp
    ]]

    df_input = pd.DataFrame(input_data, columns=feat_cols)
    delta = model.predict(df_input)
    total = course['CoursePar'].iloc[0] + delta[0]

    return {
        "predicted_total": round(float(total), 1),
        "predicted_vs_par": round(float(delta[0]), 1),
        "current_player_rating": round(float(latest_rating), 1)
    }
