from __future__ import annotations

from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_oil_prices.csv"


def load_processed_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Missing required column: date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _apply_features_mlr_justbrent(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(
        df,
        [
            "close_x",
            "open_x",
            "high_x",
            "low_x",
            "average_x",
        ],
    )

    data = df.copy()

    for lag in [1, 3, 5, 7]:
        data[f"brent_lag_{lag}"] = data["close_x"].shift(lag)

    data["brent_ma_5"] = data["close_x"].rolling(window=5).mean()
    data["brent_ma_10"] = data["close_x"].rolling(window=10).mean()
    data["target"] = data["close_x"].shift(-1)

    feature_cols = [
        "close_x",
        "open_x",
        "high_x",
        "low_x",
        "average_x",
        "brent_lag_1",
        "brent_lag_3",
        "brent_lag_5",
        "brent_lag_7",
        "brent_ma_5",
        "brent_ma_10",
    ]
    return data, feature_cols


def build_features_mlr_justbrent_full(df: pd.DataFrame):
    data, feature_cols = _apply_features_mlr_justbrent(df)
    X_full = data[feature_cols]
    return data, X_full, feature_cols


def build_features_mlr_justbrent(df: pd.DataFrame):
    data, feature_cols = _apply_features_mlr_justbrent(df)
    data_clean = data.dropna().reset_index(drop=True)
    X = data_clean[feature_cols]
    y = data_clean["target"]
    return data_clean, X, y, feature_cols


def _apply_features_nn(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(
        df,
        [
            "close_x",
            "open_x",
            "high_x",
            "low_x",
            "average_x",
            "close_y",
            "open_y",
            "high_y",
            "low_y",
            "average_y",
        ],
    )

    data = df.copy()

    for lag in [1, 3, 5, 7]:
        data[f"brent_lag_{lag}"] = data["close_x"].shift(lag)
        data[f"wti_lag_{lag}"] = data["close_y"].shift(lag)

    data["brent_ma_5"] = data["close_x"].rolling(window=5).mean()
    data["brent_ma_10"] = data["close_x"].rolling(window=10).mean()
    data["wti_ma_5"] = data["close_y"].rolling(window=5).mean()
    data["wti_ma_10"] = data["close_y"].rolling(window=10).mean()
    data["target"] = data["close_x"].shift(-1)

    feature_cols = [
        "close_x",
        "open_x",
        "high_x",
        "low_x",
        "average_x",
        "brent_lag_1",
        "brent_lag_3",
        "brent_lag_5",
        "brent_lag_7",
        "brent_ma_5",
        "brent_ma_10",
        "close_y",
        "open_y",
        "high_y",
        "low_y",
        "average_y",
        "wti_lag_1",
        "wti_lag_3",
        "wti_lag_5",
        "wti_lag_7",
        "wti_ma_5",
        "wti_ma_10",
    ]
    return data, feature_cols


def build_features_nn_full(df: pd.DataFrame):
    data, feature_cols = _apply_features_nn(df)
    X_full = data[feature_cols]
    return data, X_full, feature_cols


def build_features_nn(df: pd.DataFrame):
    data, feature_cols = _apply_features_nn(df)
    data_clean = data.dropna().reset_index(drop=True)
    X = data_clean[feature_cols]
    y = data_clean["target"]
    return data_clean, X, y, feature_cols


def _apply_features_rf(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(
        df,
        [
            "open_x",
            "high_x",
            "low_x",
            "close_x",
            "volume_x",
            "average_x",
            "open_y",
            "high_y",
            "low_y",
            "close_y",
            "volume_y",
            "average_y",
        ],
    )

    data = df.copy()

    for lag in [1, 3, 5, 7]:
        data[f"brent_close_lag_{lag}"] = data["close_x"].shift(lag)
        data[f"wti_close_lag_{lag}"] = data["close_y"].shift(lag)
        data[f"brent_volume_lag_{lag}"] = data["volume_x"].shift(lag)

    data["brent_close_ma_5"] = data["close_x"].rolling(window=5).mean()
    data["brent_close_ma_10"] = data["close_x"].rolling(window=10).mean()
    data["wti_close_ma_5"] = data["close_y"].rolling(window=5).mean()
    data["wti_close_ma_10"] = data["close_y"].rolling(window=10).mean()

    data["brent_high_low_diff"] = data["high_x"] - data["low_x"]
    data["wti_high_low_diff"] = data["high_y"] - data["low_y"]
    data["brent_open_close_diff"] = data["close_x"] - data["open_x"]
    data["brent_wti_spread"] = data["close_x"] - data["close_y"]

    data["target_brent_next_day"] = data["close_x"].shift(-1)

    feature_cols = [
        "brent_close_lag_1",
        "brent_close_lag_3",
        "brent_close_lag_5",
        "brent_close_lag_7",
        "wti_close_lag_1",
        "wti_close_lag_3",
        "wti_close_lag_5",
        "wti_close_lag_7",
        "brent_volume_lag_1",
        "brent_volume_lag_3",
        "brent_volume_lag_5",
        "brent_volume_lag_7",
        "brent_close_ma_5",
        "brent_close_ma_10",
        "wti_close_ma_5",
        "wti_close_ma_10",
        "brent_high_low_diff",
        "wti_high_low_diff",
        "brent_open_close_diff",
        "brent_wti_spread",
        "open_x",
        "high_x",
        "low_x",
        "close_x",
        "volume_x",
        "average_x",
        "open_y",
        "high_y",
        "low_y",
        "close_y",
        "volume_y",
        "average_y",
    ]
    return data, feature_cols


def build_features_rf_full(df: pd.DataFrame):
    data, feature_cols = _apply_features_rf(df)
    X_full = data[feature_cols]
    return data, X_full, feature_cols


def build_features_rf(df: pd.DataFrame):
    data, feature_cols = _apply_features_rf(df)
    data_clean = data.dropna().reset_index(drop=True)
    X = data_clean[feature_cols]
    y = data_clean["target_brent_next_day"]
    return data_clean, X, y, feature_cols


def split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_train_test(X, y, train_ratio=0.8):
    n = len(X)
    train_end = int(n * train_ratio)
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]
    return X_train, X_test, y_train, y_test
