# NFL Fantasy Points Predictor (PPR)

Predicts next-week PPR fantasy points for QB/RB/WR/TE from weekly game logs.

**Stack:** Python, pandas, NumPy, scikit-learn (preprocessing), TensorFlow/Keras (neural net),
nfl_data_py (data), Matplotlib (plots)

## How to run
pip install -r requirements.txt
python fantasy.py

## What it does
- Loads 2013–2022 for training; holds out 2024 for testing
- Cleans team/opponent/name fields
- Builds simple features (last game + 3/5-game averages)
- Trains one small model per position; reports MAE on 2024
- Writes `leaderboard.csv` with top predictions
