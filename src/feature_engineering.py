# load data/processed/matches_clean.csv
# turn each match into 2 rows:
#   1. winner row with winner as the first player
#   2. loser row with loser as the first player
#add basic features of rank_diff, age_diff, height_diff, etc.
# split by yera into train / validation / test sets
# save train.sv, val.csv, test.csv into data/processed/

import pandas as pd
import os

from src.config import (
    CLEAN_MATCHES_FILENAME,
    PLAYER_VIEW_FILENAME,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
)
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