from __future__ import annotations

from pathlib import Path
from datetime import date
import joblib
import pandas as pd
import streamlit as st
import altair as alt

from src.features import (
    load_processed_data,
    build_features_mlr_justbrent,
    build_features_mlr_justbrent_full,
    build_features_nn,
    build_features_nn_full,
    build_features_rf,
    build_features_rf_full,
    split_train_val_test,
    split_train_test,
)
from src.metrics import regression_metrics


REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"


st.set_page_config(
    page_title="Brent Crude Oil Prediction Demo",
    layout="wide",
)


@st.cache_data
def load_data():
    return load_processed_data()


@st.cache_resource
def load_model_artifacts():
    return {
        "MLR (JustBrent)": {
            "model": joblib.load(MODELS_DIR / "mlr_justbrent_model.pkl"),
            "scaler": joblib.load(MODELS_DIR / "mlr_justbrent_scaler.pkl"),
            "type": "mlr",
        },
        "Random Forest Regressor": {
            "model": joblib.load(MODELS_DIR / "rf_model.pkl"),
            "scaler": None,
            "type": "rf",
        },
        "Neural Network (MLPRegressor)": {
            "model": joblib.load(MODELS_DIR / "nn_model.pkl"),
            "scaler": joblib.load(MODELS_DIR / "nn_scaler.pkl"),
            "type": "nn",
        },
    }


def prepare_model_data(df: pd.DataFrame, model_key: str):
    if model_key == "MLR (JustBrent)":
        data_full, X_full, _ = build_features_mlr_justbrent_full(df)
        data_clean, X, y, _ = build_features_mlr_justbrent(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
        split = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
    elif model_key == "Neural Network (MLPRegressor)":
        data_full, X_full, _ = build_features_nn_full(df)
        data_clean, X, y, _ = build_features_nn(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
        split = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
    else:
        data_full, X_full, _ = build_features_rf_full(df)
        data_clean, X, y, _ = build_features_rf(df)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        split = {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
        }

    return data_full, X_full, data_clean, X, y, split


def predict_all(model_key, model_bundle, X: pd.DataFrame):
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    if scaler is not None:
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
    else:
        preds = model.predict(X)

    return preds


def section_title(text: str):
    st.markdown(f"**{text}**")


def main():
    st.title("Brent Crude Oil Prediction Dashboard")
    st.caption("Demo berbasis `data/processed/merged_oil_prices.csv` tanpa training ulang saat aplikasi berjalan.")

    if not MODELS_DIR.exists():
        st.error("Folder `models/` tidak ditemukan. Jalankan export model terlebih dahulu.")
        st.stop()

    df = load_data()
    model_artifacts = load_model_artifacts()

    st.sidebar.header("Pengaturan")
    model_key = st.sidebar.selectbox(
        "Pilih Model",
        list(model_artifacts.keys()),
        index=0,
    )

    data_full, X_full, data_clean, X, y, split = prepare_model_data(df, model_key)
    preds = predict_all(model_key, model_artifacts[model_key], X)

    dashboard_container = st.container()
    chart_container = st.container()

    # Visualisasi Harga vs Prediksi
    min_date = data_clean["date"].min().date()
    max_date = data_clean["date"].max().date()

    if "range_start" not in st.session_state:
        lower_bound = max(min_date, date(2024, 1, 1))
        default_start = (data_clean["date"].max() - pd.DateOffset(years=1)).date()
        default_start = max(default_start, lower_bound)
        if default_start > max_date:
            default_start = min_date
        st.session_state.range_start = default_start
    if "range_end" not in st.session_state:
        st.session_state.range_end = max_date

    with chart_container:
        section_title("Visualisasi Harga vs Prediksi")
        with st.form("date_range_form"):
            c1, c2 = st.columns(2)
            with c1:
                start_input = st.date_input(
                    "Tanggal Mulai",
                    value=st.session_state.range_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="range_start_input",
                )
            with c2:
                end_input = st.date_input(
                    "Tanggal Akhir",
                    value=st.session_state.range_end,
                    min_value=min_date,
                    max_value=max_date,
                    key="range_end_input",
                )
            submitted = st.form_submit_button("Apply")

        if submitted:
            if start_input > end_input:
                st.error(
                    "Tanggal mulai harus lebih kecil atau sama dengan tanggal akhir."
                )
            else:
                st.session_state.range_start = start_input
                st.session_state.range_end = end_input

        start_date = st.session_state.range_start
        end_date = st.session_state.range_end

        plot_df = pd.DataFrame(
            {
                "date": data_clean["date"],
                "actual_next_close": y.values,
                "pred_next_close": preds,
            }
        )
        plot_df = plot_df[
            (plot_df["date"].dt.date >= start_date)
            & (plot_df["date"].dt.date <= end_date)
        ]
        plot_df = plot_df.set_index("date")
        plot_long = (
            plot_df.reset_index()
            .melt(
                id_vars="date",
                value_vars=["actual_next_close", "pred_next_close"],
                var_name="series",
                value_name="value",
            )
            .copy()
        )
        if plot_long.empty:
            st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
            return
        y_min = float(plot_long["value"].min())
        y_max = float(plot_long["value"].max())
        pad = max((y_max - y_min) * 0.05, 0.5)
        y_domain = [y_min - pad, y_max + pad]
        series_label = {
            "actual_next_close": "Actual Next Close",
            "pred_next_close": "Predicted Next Close",
        }
        plot_long["series"] = plot_long["series"].map(series_label).fillna(
            plot_long["series"]
        )
        chart = (
            alt.Chart(plot_long)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Price", scale=alt.Scale(domain=y_domain)),
                color=alt.Color("series:N", title="Series"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Price", format=",.2f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Catatan: Prediksi merepresentasikan harga penutupan **hari berikutnya**."
        )

    # Dashboard Sinyal Harian Brent (Utama) - mengikuti tanggal akhir pada range
    X_full_valid = X_full.dropna()
    valid_dates = data_full.loc[X_full_valid.index, "date"]
    eligible = valid_dates[valid_dates.dt.date <= end_date]
    if eligible.empty:
        selected_idx = X_full_valid.index[0]
        selected_date = valid_dates.iloc[0].date()
        warning_msg = (
            "Tanggal akhir terlalu awal; menggunakan tanggal awal yang tersedia."
        )
    else:
        selected_idx = eligible.index[-1]
        selected_date = valid_dates.loc[selected_idx].date()
        warning_msg = None

    selected_features = X_full_valid.loc[[selected_idx]]
    selected_close = float(data_full.loc[selected_idx, "close_x"])
    selected_pred = float(
        predict_all(model_key, model_artifacts[model_key], selected_features)[0]
    )
    delta = selected_pred - selected_close
    signal = "BUY" if selected_pred > selected_close else "SELL"

    with dashboard_container:
        section_title("Dashboard Sinyal Harian Brent (Utama)")
        if warning_msg:
            st.warning(warning_msg)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Close Tanggal Akhir", f"{selected_close:,.2f}")
        kpi2.metric("Prediksi Hari Berikutnya", f"{selected_pred:,.2f}")
        kpi3.metric("Δ Prediksi", f"{delta:,.2f}")
        kpi4.metric("Sinyal", signal)
        st.caption(f"Sinyal dihitung berdasarkan data tanggal {selected_date}.")

    # Monitoring Akurasi Prediksi
    section_title("Monitoring Akurasi Prediksi")
    metrics_blocks = []

    if "val" in split:
        for split_name in ["train", "val", "test"]:
            X_split, y_split = split[split_name]
            if model_artifacts[model_key]["scaler"] is not None:
                X_split = model_artifacts[model_key]["scaler"].transform(X_split)
            y_pred = model_artifacts[model_key]["model"].predict(X_split)
            metrics_blocks.append((split_name, regression_metrics(y_split, y_pred)))
    else:
        for split_name in ["train", "test"]:
            X_split, y_split = split[split_name]
            y_pred = model_artifacts[model_key]["model"].predict(X_split)
            metrics_blocks.append((split_name, regression_metrics(y_split, y_pred)))

    cols = st.columns(len(metrics_blocks))
    for col, (name, metrics) in zip(cols, metrics_blocks):
        col.subheader(name.capitalize())
        col.write(f"MSE: {metrics['MSE']:.4f}")
        col.write(f"RMSE: {metrics['RMSE']:.4f}")
        col.write(f"MAE: {metrics['MAE']:.4f}")
        col.write(f"R²: {metrics['R2']:.4f}")

    # Model & Dokumentasi Arsitektur Sistem
    section_title("Model & Dokumentasi Arsitektur Sistem")
    st.markdown(
        """
Model yang digunakan dalam demo ini:
- MLR (JustBrent)
- Random Forest Regressor
- Neural Network (MLPRegressor)

Alur sistem (ringkas):
```
Processed CSV -> Feature Engineering -> Model -> Prediksi -> Dashboard
```

Catatan implementasi:
- MLR menggunakan fitur Brent saja (tanpa WTI).
- RF menggunakan fitur tambahan (lag volume, spread, high-low range).
- NN menggunakan fitur Brent + WTI, dengan StandardScaler.
"""
    )


if __name__ == "__main__":
    main()
