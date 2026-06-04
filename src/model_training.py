import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = ['day_of_week', 'is_weekend', 'is_rush_hour', 'is_night',
            'Traffic_Count', 'temperature', 'wind_speed', 'precipitation', 'humidity', 'pm25',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'heat_index', 'traffic_ratio', 'traffic_log', 'city_base_noise']
TARGET   = 'Noise_Level_dB'

# Paths relative to project root
DATA_PROCESSED = "data/processed_data.csv"
MODEL_PATH     = "models/noise_model.pkl"
META_PATH      = "models/model_meta.pkl"


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"{name:30s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return rmse, mae, r2, preds


def main():
    df = pd.read_csv(DATA_PROCESSED)
    X  = df[FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("=" * 60)
    print("  Model Training & Evaluation")
    print("=" * 60)

    candidates = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        rmse, mae, r2, _ = evaluate(name, model, X_test, y_test)
        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2}

    # Cross-validation on best two
    print("\n--- 5-Fold Cross-Validation (RMSE) ---")
    for name in ["Random Forest", "Gradient Boosting"]:
        cv_scores = cross_val_score(
            candidates[name], X, y,
            cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        print(f"{name:30s}  CV RMSE = {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Select best model by RMSE
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    print(f"\nBest Model: {best_name}  (RMSE={results[best_name]['rmse']:.4f}, R²={results[best_name]['r2']:.4f})")

    # Feature importance (tree models)
    if hasattr(best_model, "feature_importances_"):
        print("\n--- Feature Importances ---")
        importances = best_model.feature_importances_
        for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
            print(f"  {feat:20s}: {imp:.4f}")

    # Save model + metadata
    joblib.dump(best_model, MODEL_PATH)
    meta = {
        "best_model_name": best_name,
        "features": FEATURES,
        "metrics": {k: {"rmse": v["rmse"], "mae": v["mae"], "r2": v["r2"]}
                    for k, v in results.items()}
    }
    joblib.dump(meta, META_PATH)
    print(f"\nSaved {MODEL_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()
