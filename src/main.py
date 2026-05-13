import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Disc Golf Score Predictor")

# Load latest state
reg = pd.read_csv('data/course_registry.csv')
processed = pd.read_csv('data/processed_scores.csv')
latest_rating = processed['PlayerRating_At_Time'].iloc[-1]

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)


class PredictRequest(BaseModel):
    CourseName: str
    LayoutName: str
    Month: int
    Duration_Min: float
    Wind: float
    Temp: float


@app.post("/predict")
def predict(data: PredictRequest):
    course = reg[(reg.CourseName == data.CourseName) &
                 (reg.LayoutName == data.LayoutName)]

    if course.empty:
        raise HTTPException(status_code=404, detail="Course not found")

    # Order must match training: PlayerRating, ParRating, etc.
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
    total = course['CoursePar'].iloc[0] + delta

    return {
        "predicted_total": round(float(total[0]), 1),
        "predicted_vs_par": round(float(delta[0]), 1),
        "current_player_rating": round(latest_rating, 1)
    }
