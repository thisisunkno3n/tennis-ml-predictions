# data loading and preprocessing
import glob
import pandas as pd
import numpy as np
import os
from typing import List
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, CLEAN_MATCHES_FILENAME

def _list_files() -> list[str]:
    #List all match files in the raw data directory
    # All filenames are currently in a format of "atp_matches_YYYY.csv"
    # so we can use the year from the filename to load the data
    pattern = os.path.join(DATA_RAW_DIR, "atp_matches_*.csv")                 #sets a pattern to match all files in the raw data directory that start with "atp_matches_" and end with ".csv"
    files = sorted(glob.glob(pattern))                                      
    if not files:
        raise FileNotFoundError(f"No match files found in {DATA_RAW_DIR}")
    return files

def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:   
    #Clean the matches data

    original_len = len(df)     #original length of the dataframe

    # Remove Davis Cup
    if "tourney_level" in df.columns:    #checks if the tourney_level column exists in the dataframe and if it does, it removes the Davis Cup matches
        df = df[df["tourney_level"] != "D"].copy()
        print(f"  After removing Davis Cup: {len(df)} rows (dropped {original_len - len(df)})")
        original_len = len(df)

    # Remove incomplete matches based on score - REF, W/O, DEF, ABD
    bad_score_markers = ["RET", "W/O", "DEF", "ABD"]
    df["score"] = df["score"].astype(str)                                                  #converts the score to a string
    mask_bad = df["score"].str.contains("|".join(bad_score_markers), na=False)             #boolean mask to find rows with bad score markers
    df = df[~mask_bad]                                                                   #removes the rows with bad score markers
    print(f"  After removing incomplete matches: {len(df)} rows (dropped {original_len - len(df)})")
    original_len = len(df)

    # Convert tourney_date to datetime and add year column
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["year"] = df["tourney_date"].dt.year

    # Ensuring important columns are actually numeric
    numeric_cols = [
        "winner_rank",
        "winner_rank_points",
        "loser_rank",
        "loser_rank_points",
        "winner_age",
        "loser_age",
        "winner_ht",
        "loser_ht",
        "minutes"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") #converts to numeric type, if it fails, it converts the value to NaN


    # Drop rows with missing winner/loser rank (NaN values)
    df = df.dropna(subset=["winner_rank", "loser_rank"])
    print(f"  After dropping rows with missing rank: {len(df)} rows (dropped {original_len - len(df)})")
    original_len = len(df)

    # Drop rows with missing winner/loser rank points (NaN values)
    df = df.dropna(subset=["winner_rank_points", "loser_rank_points"])
    print(f"  After dropping rows with missing rank points: {len(df)} rows (dropped {original_len - len(df)})")
    original_len = len(df)

    df = df.reset_index(drop=True)   #resets the index to 0, 1, 2, ... after filtering
    return df    #returns the cleaned dataframe

#------------------------------ Testing Loading Files --------------------------------
#test 1: list files
# files = _list_files()
# print(f"Found {len(files)} files")
# print(files[:3])       # prints first 3 files: ['data/raw/atp_matches_2013.csv', 'data/raw/atp_matches_2014.csv', 'data/raw/atp_matches_2015.csv']


#------------------------------ Testing Davis Cup Removal --------------------------------
if __name__ == "__main__":
    # Test with one file
    test_file = "data/raw/atp_matches_2013.csv"
    df_raw = pd.read_csv(test_file)
    print(f"Original rows: {len(df_raw)}")
    
    # Check Davis Cup before
    davis_cup_count = len(df_raw[df_raw["tourney_level"] == "D"])
    print(f"Davis Cup matches before: {davis_cup_count}")
    
    # Clean it
    df_clean = _clean_matches(df_raw)
    
    # Verify Davis Cup removed
    davis_cup_after = len(df_clean[df_clean["tourney_level"] == "D"])
    print(f"Davis Cup matches after: {davis_cup_after}")
    print(f"Final rows: {len(df_clean)}")