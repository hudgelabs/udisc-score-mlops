import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Disc Golf Score Predictor")

model = None
encoder = None

# Check if files exist to prevent CI import errors
if os.path.exists('models/model.pkl') and os.path.exists('models/encoder.pkl'):
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)


class RoundRequest(BaseModel):
    CourseName: str
    Month: int
    HourOfDay: int
    Duration_Min: float


@app.get("/")
def health_check():
    return {"status": "Online", "model_loaded": model is not None}


@app.post("/predict")
def predict_score(data: RoundRequest):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_df = pd.DataFrame([data.dict()])
        input_df['CourseName'] = encoder.transform(input_df['CourseName'])
        prediction = model.predict(input_df)
        return {
            "input": data.dict(),
            "predicted_total_score": round(float(prediction[0]), 2)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Course not recognized")
