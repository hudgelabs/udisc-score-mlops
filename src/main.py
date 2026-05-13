from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import os
from pydantic import BaseModel

app = FastAPI()

# Load Latest State
reg = pd.read_csv('data/course_registry.csv')
processed = pd.read_csv('data/processed_scores.csv')
latest_rating = processed['PlayerRating_At_Time'].iloc[-1]

# Load Model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictRequest(BaseModel):
    CourseName: str
    LayoutName: str
    Wind: float
    Temp: float
    Month: int
    Duration_Min: float

@app.post("/predict")
def predict(data: PredictRequest):
    course = reg[(reg.CourseName == data.CourseName) & (reg.LayoutName == data.LayoutName)]
    if course.empty:
        raise HTTPException(status_code=404, detail="Course metadata missing from registry")

    # Features MUST match the order in preprocess.py
    # ['PlayerRating_At_Time', 'ParRating', 'GlobalAvgScore', 'CoursePar', 'Month', 'Duration_Min', 'Wind', 'Temp']
    features = [[
        latest_rating,
        course.ParRating.iloc[0],
        course.GlobalAvgScore.iloc[0],
        course.CoursePar.iloc[0],
        data.Month,
        data.Duration_Min,
        data.Wind,
        data.Temp
    ]]

    delta = model.predict(pd.DataFrame(features, columns=[
        'PlayerRating_At_Time', 'ParRating', 'GlobalAvgScore',
        'CoursePar', 'Month', 'Duration_Min', 'Wind', 'Temp'
    ]))[0]

    total = course.CoursePar.iloc[0] + delta

    return {
        "course": data.CourseName,
        "current_player_rating": round(latest_rating, 1),
        "predicted_total": round(total, 1),
        "vs_course_par": round(delta, 1),
        "vs_official_par": round(total - course.ParRating.iloc[0] if 'ParRating' in course else 0, 1)
    }
