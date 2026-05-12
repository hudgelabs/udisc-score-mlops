import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import pickle
import sys

def train(input_path, model_path):
    df = pd.read_csv(input_path)
    
    # 1. Handle categorical data (CourseName)
    le = LabelEncoder()
    df['CourseName'] = le.fit_transform(df['CourseName'])
    
    # Save the encoder so we can use it for future predictions
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # 2. Define Features and Target
    X = df.drop(columns=['Total'])
    y = df['Total']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Start MLflow Experiment
    mlflow.set_experiment("UDisc_Score_Prediction")
    
    with mlflow.start_run():
        # Parameters
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)
        
        # Model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Metrics
        score = model.score(X_test, y_test)
        mlflow.log_metric("r2_score", score)
        print(f"Model R2 Score: {score}")
        
        # Save Model to MLflow and local disk
        mlflow.sklearn.log_model(model, "model")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])