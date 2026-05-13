import pandas as pd
import sys
import os


def calculate_player_rating(series):
    ratings = []
    for i in range(len(series)):
        window = series.iloc[max(0, i-20):i]
        if len(window) < 3:
            ratings.append(None)
        else:
            num = max(1, int(len(window) * 0.4)) if len(window) < 20 else 8
            ratings.append(window.nlargest(num).mean())
    return ratings


def preprocess(score_path, reg_path, out_path):
    # 1. Load Data
    scores = pd.read_csv(score_path)
    reg = pd.read_csv(reg_path)

    # 2. Clean column names (handling Excel escape chars)
    scores.columns = [
        c.replace('=`', '').replace('`', '').replace('=', '').strip()
        for c in scores.columns
    ]

    # 3. Merge Registry Data
    df = pd.merge(scores, reg, on=['CourseName', 'LayoutName'], how='left')

    # 4. Time & Duration Logic
    fmt = '%Y-%m-%d %H%M'
    df['StartDate'] = pd.to_datetime(df['StartDate'], format=fmt)
    df['EndDate'] = pd.to_datetime(df['EndDate'], format=fmt)
    df['Duration_Min'] = (df['EndDate'] - df['StartDate']).dt.total_seconds() / 60
    df['Month'] = df['StartDate'].dt.month

    # 5. Skill Engine: Calculate PlayerRating (Rolling)
    df = df.sort_values('StartDate')
    df['PlayerRating_At_Time'] = calculate_player_rating(df['RoundRating'])

    # 6. Target Variable: Score vs CoursePar
    df['ScoreDelta'] = df['Total'] - df['CoursePar']

    # 7. Final Feature Selection
    features = [
        'PlayerRating_At_Time', 'ParRating', 'GlobalAvgScore',
        'CoursePar', 'Month', 'Duration_Min', 'Wind', 'Temp', 'ScoreDelta'
    ]

    clean_df = df[features].dropna()

    # 8. Save (Fixed variable name to out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    print(f"Processed {len(clean_df)} rounds.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/preprocess.py <scores> <reg> <out>")
    else:
        preprocess(sys.argv[1], sys.argv[2], sys.argv[3])
