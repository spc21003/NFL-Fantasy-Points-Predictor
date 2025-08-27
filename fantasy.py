import pandas as pd
import numpy as np
import nfl_data_py as nfl

seasons_train = list(range(2013, 2023))
season_valid = [2024]
scoring = "ppr"

# 1) include alternates so they aren't dropped in load_weekly
use_cols = [
    "player_id", "player_name", "player_display_name",
    "position", "team", "recent_team",
    "opponent", "opponent_team",
    "season", "week", "home_away",
    "fantasy_points", "fantasy_points_ppr", "receptions",
    "passing_yards", "passing_tds", "passing_int",
    "rushing_yards", "rushing_tds", "carries",
    "receiving_yards", "receiving_tds", "targets", "fumbles_lost", "sacks"
]

def load_weekly(years):
    df = nfl.import_weekly_data(years, downcast=True)
    keep = [c for c in use_cols if c in df.columns]
    return df[keep].copy()

train_df = load_weekly(seasons_train)
valid_df = load_weekly(season_valid)
df = pd.concat([train_df.assign(split="train"), valid_df.assign(split="valid")], ignore_index=True)

if "recent_team" in df.columns:
    if "team" in df.columns:
        df["team"] = df["team"].fillna(df["recent_team"])
    else:
        df = df.rename(columns={"recent_team": "team"})

if "opponent_team" in df.columns:
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].fillna(df["opponent_team"])
    else:
        df = df.rename(columns={"opponent_team": "opponent"})

if "player_name" not in df.columns and "player_display_name" in df.columns:
    df = df.rename(columns={"player_display_name": "player_name"})
elif "player_display_name" in df.columns and "player_name" in df.columns:
    df["player_name"] = df["player_name"].fillna(df["player_display_name"])


df["y"] = df["fantasy_points_ppr"]

df = df.sort_values(["player_id", "season", "week"])

roll_base = ["receptions", "targets", "carries",
             "rushing_yards", "rushing_tds",
             "receiving_yards", "receiving_tds", 
             "passing_yards", "passing_tds", "passing_int"]

present_cols = [c for c in roll_base if c in df.columns]

for c in present_cols:
    df[f"{c}_L3_mean"]  = df.groupby("player_id")[c].shift(1).rolling(3, min_periods=1).mean()
    df[f"{c}_L5_mean"]  = df.groupby("player_id")[c].shift(1).rolling(5, min_periods=1).mean()
    df[f"{c}_prev"]     = df.groupby("player_id")[c].shift(1)

df["games_played_prev"] = df.groupby("player_id").cumcount()

df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_feats = [c for c in df.columns if any(k in c for k in ["_L3_", "_L5_", "_prev"]) or c=="games_played_prev"]
cat_feats = [c for c in ["team", "opponent", "position", "home_away"] if c in df.columns]

def build_xy(frame):
    x_raw = frame[num_feats + cat_feats].copy()
    y = frame["y"].astype("float32").values

    num_ix = list(range(len(num_feats)))
    cat_ix = list(range(len(num_feats), len(num_feats) + len(cat_feats)))

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_ix),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_ix)
    ])
    X = pre.fit_transform(x_raw)
    return X.astype("float32"), y.astype("float32"), pre

frames = {pos: df[df["position"] ==pos] for pos in ["QB", "RB", "WR", "TE"]}
splits = {}
for pos,frame in frames.items():
    train = frame[frame["split"]=="train"]
    valid = frame[frame["split"]=="valid"]
    Xtr, ytr, pre = build_xy(train)
    Xva = pre.transform(valid[num_feats + cat_feats]).astype("float32")
    yva = valid["y"].astype("float32").values
    splits[pos] = (Xtr, ytr, Xva, yva, pre, valid.index)



import tensorflow as tf
from tensorflow import keras

def make_model(input_dim):
    return keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation = "relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation = "relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

models = {}
histories = {}
for pos, (Xtr, ytr, Xva, yva, pre, valid_idx) in splits.items():
    model = make_model(Xtr.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mae", metrics=["mae","mse"])
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_mae")
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                  epochs=200, batch_size=256, callbacks=[es], verbose=0)
    models[pos] = (model, pre)
    histories[pos] = h.history

def evaluate_position(pos):
    model, pre = models[pos]
    frame = frames[pos]
    valid = frame[frame["split"]=="valid"]
    Xva = pre.transform(valid[num_feats + cat_feats]).astype("float32")
    preds = model.predict(Xva, verbose=0).ravel()
    valid = valid.assign(pred=preds, err=np.abs(preds - valid["y"].values))
    mae = valid["err"].mean()
    return mae, valid.sort_values("pred", ascending=False)

for pos in ["QB", "RB", "WR", "TE"]:
    mae, leaderboard = evaluate_position(pos)
    print(pos, "Validation MAE:", round(mae,2))
    print(leaderboard[["season", "week", "player_name", "team", "y", "pred"]].head(10))