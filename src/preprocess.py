import pandas as pd
import sys
import os


def calculate_player_rating(series):
    ratings = []
    for i in range(len(series)):
        window = series.iloc[max(0, i-20):i]
        if len(window) < 3:  # Need history to start
            ratings.append(None)
        else:
            n = max(1, int(len(window) * 0.4)) if len(window) < 20 else 8
            ratings.append(window.nlargest(n).mean())
    return ratings


def preprocess(score_path, reg_path, out_path):
    scores = pd.read_csv(score_path)
    reg = pd.read_csv(reg_path)

    # Standardize column names
    scores.columns = [c.replace('=`', '').replace('`', '') for c in scores.columns]
    scores.columns = [c.replace('=', '').strip() for c in scores.columns]

    df = pd.merge(scores, reg, on=['CourseName', 'LayoutName'], how='left')

    fmt = '%Y-%m-%d %H%M'
    df['StartDate'] = pd.to_datetime(df['StartDate'], format=fmt)
    df['EndDate'] = pd.to_datetime(df['EndDate'], format=fmt)
    df['Duration_Min'] = (df['EndDate'] - df['StartDate']).dt.total_seconds() / 60
    df['Month'] = df['StartDate'].dt.month

    df = df.sort_values('StartDate')
    df['PlayerRating_At_Time'] = calculate_player_rating(df['RoundRating'])

    df['ScoreDelta'] = df['Total'] - df['CoursePar']

    features = [
        'PlayerRating_At_Time', 'ParRating', 'GlobalAvgScore',
        'CoursePar', 'Month', 'Duration_Min', 'Wind', 'Temp', 'ScoreDelta'
    ]

    clean_df = df[features].dropna()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"Processed {len(clean_df)} rounds.")


if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2], sys.argv[3])
