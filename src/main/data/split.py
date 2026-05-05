from __future__ import annotations

import pandas as pd


def split_dataframe(
    df: pd.DataFrame,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_split + val_split + test_split) - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df
