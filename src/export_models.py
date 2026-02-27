from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.features import (
    load_processed_data,
    build_features_mlr_justbrent,
    build_features_nn,
    build_features_rf,
    split_train_val_test,
    split_train_test,
)
from src.metrics import regression_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _save_meta(name: str, feature_cols, split_sizes, metrics):
    payload = {
        "model": name,
        "feature_cols": feature_cols,
        "split_sizes": split_sizes,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODELS_DIR / f"{name}_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def train_mlr_justbrent(df):
    data_clean, X, y, feature_cols = build_features_mlr_justbrent(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    train_metrics = regression_metrics(y_train, model.predict(X_train_scaled))
    val_metrics = regression_metrics(y_val, model.predict(X_val_scaled))
    test_metrics = regression_metrics(y_test, model.predict(X_test_scaled))

    joblib.dump(model, MODELS_DIR / "mlr_justbrent_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "mlr_justbrent_scaler.pkl")

    split_sizes = {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test),
    }
    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    _save_meta("mlr_justbrent", feature_cols, split_sizes, metrics)


def train_nn(df):
    data_clean, X, y, feature_cols = build_features_nn(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        verbose=False,
    )
    model.fit(X_train_scaled, y_train)

    train_metrics = regression_metrics(y_train, model.predict(X_train_scaled))
    val_metrics = regression_metrics(y_val, model.predict(X_val_scaled))
    test_metrics = regression_metrics(y_test, model.predict(X_test_scaled))

    joblib.dump(model, MODELS_DIR / "nn_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "nn_scaler.pkl")

    split_sizes = {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test),
    }
    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    _save_meta("nn", feature_cols, split_sizes, metrics)


def train_rf(df):
    data_clean, X, y, feature_cols = build_features_rf(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_metrics = regression_metrics(y_train, model.predict(X_train))
    test_metrics = regression_metrics(y_test, model.predict(X_test))

    joblib.dump(model, MODELS_DIR / "rf_model.pkl")

    split_sizes = {
        "train": len(X_train),
        "test": len(X_test),
    }
    metrics = {
        "train": train_metrics,
        "test": test_metrics,
    }
    _save_meta("rf", feature_cols, split_sizes, metrics)


def main():
    df = load_processed_data()
    train_mlr_justbrent(df)
    train_rf(df)
    train_nn(df)
    print("Models exported to:", MODELS_DIR)


if __name__ == "__main__":
    main()
