# temp file for testing missing values
import pandas as pd

df = pd.read_csv("data/processed/matches_clean.csv")

print(f"Total rows: {len(df)}")
print(f"\nMissing values in age columns:")
print(f"  winner_age: {df['winner_age'].isna().sum()}")
print(f"  loser_age: {df['loser_age'].isna().sum()}")

print(f"\nMissing values in all columns:")
print(df.isna().sum().to_string())

