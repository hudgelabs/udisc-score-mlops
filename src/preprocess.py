import pandas as pd
import sys
import os


def preprocess(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # 1. Cleaning: Drop incomplete rounds
    df = df.dropna(subset=['Total', 'StartDate', 'EndDate'])

    # 2. Convert to Datetime objects
    fmt = '%Y-%m-%d %H%M'
    df['StartDate'] = pd.to_datetime(df['StartDate'], format=fmt)
    df['EndDate'] = pd.to_datetime(df['EndDate'], format=fmt)

    # 3. Feature Engineering: Round Duration
    duration = (df['EndDate'] - df['StartDate']).dt.total_seconds() / 60
    df['Duration_Min'] = duration

    # 4. Feature Engineering: Time-based features
    df['Month'] = df['StartDate'].dt.month
    df['HourOfDay'] = df['StartDate'].dt.hour

    # 5. Select features for the MLOps pipeline
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
