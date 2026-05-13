# 🥏 UDisc Score Prediction Pipeline (MLOps)

An end-to-end MLOps architecture that predicts disc golf performance using historical UDisc data. This project demonstrates a transition from a simple score logger to a **Generalized Skill Engine** capable of predicting scores for new, unplayed courses with high accuracy.

## 📊 Performance Milestone
* **Model Accuracy:** **0.96 $R^2$ Score**
* **The Breakthrough:** By shifting the target variable to `ScoreDelta` (Total - CoursePar) and implementing a rolling **PlayerRating** (Top 8 of 20 rounds), the model achieved a significant improvement in predictive power over the baseline.

## 🛠 Tech Stack
* **Version Control:** **Git** for code; **DVC** (Data Version Control) for dataset and model versioning.
* **Experiment Tracking:** **MLflow** for hyperparameter logging and performance metrics.
* **API Framework:** **FastAPI** with CI-safe logic for local and cloud inference.
* **Containerization:** **Docker** for portable, cross-platform deployment.
* **CI/CD:** **GitHub Actions** for automated linting and unit testing.
* **Model:** **Scikit-learn** (Random Forest Regressor) with weather-aware features.

## 🚀 Key Features
*   **Generalized Prediction:** Predicts scores for new courses by analyzing the relationship between player skill and course difficulty (`ParRating` and `CoursePar`).
*   **Weather-Aware:** Integrates manual **Wind** and **Temperature** data to adjust predictions based on environmental variables.
*   **Dynamic Skill Engine:** Replicates the official UDisc Player Rating logic to provide a real-time "form" metric for every prediction.
*   **Static Feature Store:** Uses a `course_registry` lookup table to automatically hydrate API requests with complex course metadata.

## 📈 How it Works
1.  **DVC Pipeline:** A multi-stage pipeline (`dvc.yaml`) ensures that any change to the raw CSV or course registry automatically triggers a clean re-train.
2.  **Model Training:** Features include rolling PlayerRating, Global Average Score, Wind, Temp, and Month.
3.  **Automated Quality:** Every `git push` triggers a **GitHub Action** that validates API health and lints code for PEP 8 compliance.

## 💻 Quick Start

### Local Setup
```powershell
# Install dependencies
pip install -r requirements.txt

# Reproduce the pipeline
dvc repro

# Launch the API
uvicorn src.main:app --reload
```

### Docker Deployment
```powershell
# Build the image
docker build -t disc-golf-predictor .

# Run the container
docker run -p 8000:8000 disc-golf-predictor
```
