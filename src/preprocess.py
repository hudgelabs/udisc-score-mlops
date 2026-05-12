import pandas as pd
import sys
import os

def preprocess(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    
    # 1. Cleaning: Ensure we have the necessary columns and drop incomplete rounds
    # UDisc often uses 'Total' for the final score
    df = df.dropna(subset=['Total', 'StartDate', 'EndDate'])
    
    # 2. Convert to Datetime objects
    # The format '%Y-%m-%d %H%M' matches '2026-05-02 1941'
    df['StartDate'] = pd.to_datetime(df['StartDate'], format='%Y-%m-%d %H%M')
    df['EndDate'] = pd.to_datetime(df['EndDate'], format='%Y-%m-%d %H%M')
    
    # 3. Feature Engineering: Round Duration
    # Does a 4-hour round lead to worse scores than a 2-hour round?
    df['Duration_Min'] = (df['EndDate'] - df['StartDate']).dt.total_seconds() / 60
    
    # 4. Feature Engineering: Time-based features
    df['Month'] = df['StartDate'].dt.month
    df['HourOfDay'] = df['StartDate'].dt.hour
    
    # 5. Select features for the MLOps pipeline
    # We'll predict 'Total' using these inputs
    features = ['CourseName', 'Month', 'HourOfDay', 'Duration_Min', 'Total']
    clean_df = df[features]
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"Processed data with {len(clean_df)} rounds saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/preprocess.py <input_path> <output_path>")
    else:
        preprocess(sys.argv[1], sys.argv[2])