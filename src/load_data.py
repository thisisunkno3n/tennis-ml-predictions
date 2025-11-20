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

# def load_and_clean_matches():
#     files = _list_files()
#     cleaned_dfs = []
#     for file in files:
#         print(f"Loading {file}")
#         #Load SINGLE file
#         df = pd.read_csv(file)
#         #Clean the data
#         df_clean = _clean_matches(df)
#         #Append to the cleaned dfs
#         cleaned_dfs.append(df_clean)
#     #Concatenate all cleaned dfs into a single DataFrame
#     all_cleaned_matches = pd.concat(cleaned_dfs, ignore_index=True)     #ignore_index=True is to avoid duplicate indices by setting them to (0, 1, 2, ...)
#     return all_cleaned_matches



#------------------------------ Testing Code --------------------------------
#test 1: list files
files = _list_files()
print(f"Found {len(files)} files")
print(files[:3])

#Test step 2 - load one file
