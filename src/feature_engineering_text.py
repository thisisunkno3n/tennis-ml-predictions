# Feature engineering for text-based models (DeBERTa, etc.)
# Keeps identity features for text serialization

import pandas as pd
from src.feature_engineering import (
    _load_clean_matches,
    _build_player_view,
    _extract_date_features,
    _add_basic_features,
    _split_by_year,
    save_to_csv,
)


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
    train_text, val_text, test_text = run_feature_engineering_for_text()
    print(f"\nText dataset:")
    print(f"Train rows: {len(train_text)}, Val rows: {len(val_text)}, Test rows: {len(test_text)}")

