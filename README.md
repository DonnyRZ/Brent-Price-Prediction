# 🛢️ Brent Crude Oil Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20Pandas-orange)

A comparative study of machine learning models for predicting **Brent Crude Oil Futures** prices. This project is part of the **Fundamentals of Artificial Intelligence** course.

## 📌 Project Overview
This project aims to forecast daily crude oil prices using historical data. We implement and compare the performance of three different regression models to understand their strengths and limitations in time-series forecasting.

### 🧠 Models Implemented
1.  **Multiple Linear Regression (MLR):** Serves as a baseline model to establish linear relationships between lagged prices.
2.  **Random Forest Regression (RFR):** Handles non-linear patterns and offers insights into feature importance.
3.  **Neural Network (NN):** Utilizes Deep Learning to capture complex, high-dimensional patterns in the market data.

## 📂 Dataset
* **Source:** [Kaggle - Historical Crude Oil Futures Prices](https://www.kaggle.com/datasets/nikitamanaenkov/historical-crude-oil-futures-prices-wti-and-brent)
* **Data Used:** Brent Crude Oil Prices (`brent_prices.csv`) & WTI Prices (`wti_prices.csv`) as a supporting feature.
* **Timeframe:** 1987 - Present (Data is preprocessed to handle date alignment between markets).

## ⚙️ Installation & Setup

To run this project locally, please follow these steps:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/K400000/Oil-Price-Prediction
    cd Oil-Price-Prediction
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

**Data Preparation:**
    Run the preprocessing script to merge and clean the data.
    ```bash
    python src/data_preprocessing.py
    ```


## 👥 Team Members
* **Mr. Supasin Khamphayae** - [GitHub Profile](https://github.com/K400000)
* **Mr. Piyakorn Kacharnont** - [GitHub Profile](https://github.com/Shiseko)
* **Mr. Jakkapat Srimek** - [GitHub Profile](https://github.com/jakkapat1513)
* **Mr. Kritamet Manakit** - [GitHub Profile](https://github.com/kritametm-gif)
* **Mr. Kittichai Kuljaruhiran** - [GitHub Profile](https://github.com/kitti1223)
* **Mr. Mawin Boonsri** - [GitHub Profile](https://github.com/Mawinbosri)
