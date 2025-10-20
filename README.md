# 📈 ML CAPE Return Predictor

A machine learning pipeline that predicts **10-year annualized real returns** from the **Cyclically Adjusted Price-to-Earnings (CAPE)** ratio.

This project combines financial theory with modern ML techniques — ideal for quantitative finance, research, or investment analysis portfolios.

---

## 🧠 Project Overview
The goal is to explore how equity market valuations (via CAPE ratio) relate to subsequent long-term real returns.
We use regression models and time-series-aware validation to produce interpretable forecasts and feature insights.

---

## ⚙️ Features
- Modular, production-style **Python 3.8+** code
- Robust **feature engineering**: log transforms, lags, rolling means/stds, percent changes
- **Models**: Ridge Regression, Random Forest, XGBoost, LightGBM (auto-detected if installed)
- **TimeSeriesSplit** cross-validation for realistic temporal evaluation
- **GridSearchCV** for hyperparameter tuning
- Visual outputs:
  - `outputs/prediction_plot.png` – predicted vs actual returns
  - `outputs/feature_importances.png` – top explanatory features
- Saves trained model as `best_model.joblib` for reuse

---

## 🚀 How to Run
```bash
# 1. Install dependencies
pip install pandas numpy matplotlib scikit-learn joblib xgboost lightgbm

# 2. Run the training script
python ml_cape_return_predictor.py --cape cape_ratio.csv --returns tya_real_return.csv --out model.joblib
