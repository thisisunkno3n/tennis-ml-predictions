# load data/processed/matches_clean.csv
# turn each match into 2 rows:
#   1. winner row with winner as the first player
#   2. loser row with loser as the first player
# add basic features of rank_diff, age_diff, height_diff, etc.
# split by yera into train / validation / test sets
# save train.sv, val.csv, test.csv into data/processed/

import pandas as pd
import os
import numpy as np
from typing import List
from src.config import DATA_PROCESSED_DIR
from src.config import (
    CLEAN_MATCHES_FILENAME,
    PLAYER_VIEW_FILENAME,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
)
# Identity columns to remove for tabular models (but keep for text models)
IDENTITY_COLUMNS = [
    "player_id",
    "player_name",
    "opponent_id",
    "opponent_name",
    "tourney_id",
    "tourney_name",
    # "tourney_date" - drop after extracting date features
]

def _load_clean_matches() -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED_DIR, CLEAN_MATCHES_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean matches file not found at {path}")
    df = pd.read_csv(path)
    return df

def _build_player_view(df: pd.DataFrame) -> pd.DataFrame:
    # Convert winner / loser match rows into 'player vs oppenent' rows
    # Row A: player = winner, opponent = loser, label = 1 (win)
    # Row B: player = loser, opponent = winner, label = 0 (loss)
    
    rows = []

    for _, row in df.iterrows():
        common = {
            "tourney_id": row["tourney_id"],
            "tourney_name": row["tourney_name"],
            "surface": row["surface"],
            "tourney_level": row["tourney_level"],
            "tourney_date": row["tourney_date"],
            "year": row["year"],
            "round": row["round"],
            "best_of": row["best_of"],
        }
        rows.append({
            **common,
            "player_id": row["winner_id"],
            "player_name": row["winner_name"],
            "player_hand": row["winner_hand"],
            "player_ht": row["winner_ht"],
            "player_ioc": row["winner_ioc"],
            "player_age": row["winner_age"],
            "player_rank": row["winner_rank"],
            "player_rank_points": row["winner_rank_points"],

            "opponent_id": row["loser_id"],
            "opponent_name": row["loser_name"],
            "opponent_hand": row["loser_hand"],
            "opponent_ht": row["loser_ht"],
            "opponent_ioc": row["loser_ioc"],
            "opponent_age": row["loser_age"],
            "opponent_rank": row["loser_rank"],
            "opponent_rank_points": row["loser_rank_points"],

            "label": 1,
        })
        rows.append({
            **common,
            "player_id": row["loser_id"],
            "player_name": row["loser_name"],
            "player_hand": row["loser_hand"],
            "player_ht": row["loser_ht"],
            "player_ioc": row["loser_ioc"],
            "player_age": row["loser_age"],
            "player_rank": row["loser_rank"],
            "player_rank_points": row["loser_rank_points"],

            "opponent_id": row["winner_id"],
            "opponent_name": row["winner_name"],
            "opponent_hand": row["winner_hand"],
            "opponent_ht": row["winner_ht"],
            "opponent_ioc": row["winner_ioc"],
            "opponent_age": row["winner_age"],
            "opponent_rank": row["winner_rank"],
            "opponent_rank_points": row["winner_rank_points"],

            "label": 0,
        })
    player_df = pd.DataFrame(rows)

    # Double checking numeric cols again
    numeric_cols = [
        "player_rank", "player_rank_points",
        "opponent_rank", "opponent_rank_points",
        "player_age", "opponent_age",
        "player_ht", "opponent_ht",
    ]
    for col in numeric_cols:
        if col in player_df.columns:
            player_df[col] = pd.to_numeric(player_df[col], errors="coerce")
    return player_df

def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add basic features of rank_diff, age_diff, height_diff, etc.

    df["rank_diff"] = df["player_rank"] - df["opponent_rank"]
    df["age_diff"] = df["player_age"] - df["opponent_age"]

    df["height_diff"] = df["player_ht"] - df["opponent_ht"]
    df["rank_points_diff"] = df["player_rank_points"] - df["opponent_rank_points"]
    return df

def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract useful date features before dropping tourney_date."""
    if "tourney_date" in df.columns:
        # Ensure tourney_date is datetime
        df["tourney_date"] = pd.to_datetime(df["tourney_date"])
        df["year"] = df["tourney_date"].dt.year
        df["month"] = df["tourney_date"].dt.month
        # Optional: add season features later (is_clay_season, etc.)
    return df

def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    
    categorical_cols = ["surface", "tourney_level"]  # "round" optional per spec
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])  # Drop original categorical column
    
    return df

def _remove_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that leak identity/metadata and encourage memorization
    instead of generalization. This is only for the tabular model input.
    """
    cols_to_drop = [c for c in IDENTITY_COLUMNS if c in df.columns]
    
    # Also drop tourney_date if date features already extracted
    if "tourney_date" in df.columns and "year" in df.columns:
        cols_to_drop.append("tourney_date")
    
    return df.drop(columns=cols_to_drop)

def build_tabular_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset for tabular models (LightGBM/XGBoost).
    Removes identity features to ensure generalization.
    """
    df = df.copy()
    
    # 1. Extract date features first (before dropping tourney_date)
    df = _extract_date_features(df)
    
    # 2. Add basic features (rank_diff, age_diff, etc.)
    df = _add_basic_features(df)
    
    # 3. One-hot encode categoricals
    df = _one_hot_encode(df)
    
    # 4. Remove identity features (after extracting what we need)
    df = _remove_identity_features(df)
    
    return df

def build_text_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset for DeBERTa text model.
    Keeps identity features (names, tournament info) for text serialization.
    """
    df = df.copy()
    
    # Extract date features (but keep tourney_date for text)
    df = _extract_date_features(df)
    
    # Add basic features
    df = _add_basic_features(df)
    
    # Keep all identity features for text generation
    # Don't call _remove_identity_features() here
    # Don't one-hot encode (text model will handle it in the prompt)
    
    return df

def _split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].isin(TRAIN_YEARS)]
    val = df[df["year"].isin(VAL_YEARS)]
    test = df[df["year"].isin(TEST_YEARS)]
    return train, val, test

def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(DATA_PROCESSED_DIR, filename)
    df.to_csv(path, index=False)

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save train/val/test splits to CSV files."""
    save_to_csv(train, "train.csv")
    save_to_csv(val, "val.csv")
    save_to_csv(test, "test.csv")
    print(f"Saved splits: train={len(train)}, val={len(val)}, test={len(test)}")

def run_feature_engineering() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _load_clean_matches()
    df = _build_player_view(df)
    
    # Build tabular dataset (removes identity)
    df_tabular = build_tabular_dataset(df)
    
    # Split and save
    train, val, test = _split_by_year(df_tabular)
    save_splits(train, val, test)
    
    return train, val, test

def run_feature_engineering_for_text() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pipeline for DeBERTa text model (keeps identity features)."""
    df = _load_clean_matches()
    df = _build_player_view(df)
    
    # Build text dataset (keeps identity)
    df_text = build_text_dataset(df)
    
    # Split (same time-based split)
    train, val, test = _split_by_year(df_text)
    
    # Save with different filenames to distinguish
    save_to_csv(train, "train_text.csv")
    save_to_csv(val, "val_text.csv")
    save_to_csv(test, "test_text.csv")
    print(f"Saved text splits: train={len(train)}, val={len(val)}, test={len(test)}")
    
    return train, val, test

if __name__ == "__main__":
    # Run tabular pipeline
    train, val, test = run_feature_engineering()
    print(f"\nTabular dataset:")
    print(f"Train rows: {len(train)}, Val rows: {len(val)}, Test rows: {len(test)}")
    print(f"Tabular features: {len(train.columns)} columns")
    
    # Optionally run text pipeline (uncomment when ready)
    # train_text, val_text, test_text = run_feature_engineering_for_text()
    # print(f"\nText dataset:")
    # print(f"Train rows: {len(train_text)}, Val rows: {len(val_text)}, Test rows: {len(test_text)}")