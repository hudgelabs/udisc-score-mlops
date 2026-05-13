import pandas as pd
import sys
import os

def calculate_player_rating(series):
    """Calculates rolling PlayerRating: top 8 of last 20 rounds."""
    ratings = []
    for i in range(len(series)):
        # Look back at previous rounds
        window = series.iloc[max(0, i-20):i]
        if len(window) < 3: # Need at least 3 rounds to start a 'rating'
            ratings.append(None)
        else:
            num_to_avg = max(1, int(len(window) * 0.4)) if len(window) < 20 else 8
            ratings.append(window.nlargest(num_to_avg).mean())
    return ratings

def preprocess(score_path, reg_path, out_path):
    # Try reading with comma, if it fails to find columns, try tab
    scores = pd.read_csv(score_path)
    if 'CourseName' not in scores.columns:
        scores = pd.read_csv(score_path, sep='\t')

    reg = pd.read_csv(reg_path) # Registry is definitely comma-separated

    # Standardize column names (fixing the `=+/-` and other symbols)
    scores.columns = [c.replace('=`', '').replace('`', '').replace('=', '').strip()
                      for c in scores.columns]

    # Ensure registry columns are also stripped of whitespace
    reg.columns = [c.strip() for c in reg.columns]

    # Now the merge should work
    df = pd.merge(scores, reg, on=['CourseName', 'LayoutName'], how='left')

    # 3. Time & Duration
    df['StartDate'] = pd.to_datetime(df['StartDate'], format='%Y-%m-%d %H%M')
    df = df.sort_values('StartDate')
    df['Duration_Min'] = (pd.to_datetime(df['EndDate'], format='%Y-%m-%d %H%M') - df['StartDate']).dt.total_seconds() / 60
    df['Month'] = df['StartDate'].dt.month

    # 4. Skill Engine: Calculate PlayerRating (Rolling)
    df['PlayerRating_At_Time'] = calculate_player_rating(df['RoundRating'])

    # 5. Target Variable: ScoreDelta
    df['ScoreDelta'] = df['Total'] - df['CoursePar']

    # 6. Final Feature Selection
    # Note: We include Wind and Temp as they are in your header!
    features = [
        'PlayerRating_At_Time', 'ParRating', 'GlobalAvgScore',
        'CoursePar', 'Month', 'Duration_Min', 'Wind', 'Temp', 'ScoreDelta'
    ]

    clean_df = df[features].dropna()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    print(f"Processed {len(clean_df)} rounds.")

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2], sys.argv[3])
