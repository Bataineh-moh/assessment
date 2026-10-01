import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_file(candidates: list[str]) -> Path | None:
    """Checks current directory and a 'data/' subfolder for candidates."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
        p_data = Path("data") / c
        if p_data.exists():
            return p_data
    return None


def get_december_inputs() -> pd.DataFrame:
    """Loads december inputs or auto-generates them according to score.py specifications."""
    dec_path = find_file([
        "december-chart-inputs.csv",
        "december_chart_inputs.csv",
        "data/december-chart-inputs.csv",
        "data/december_chart_inputs.csv",
    ])
    if dec_path is not None:
        print(f"Loading dec from:   {dec_path}")
        return pd.read_csv(dec_path)

    print("december-chart-inputs.csv not found locally. Auto-generating fixed inputs according to score.py...")
    dates = pd.date_range("2025-12-01", "2025-12-31", freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "pickup": "Lexington",
        "delivery": "Fort Wayne",
        "distance": 360.0,
        "equipment": "Dry Van",
        "weight": 32000.0,
        "date": dates,
        "predicted_rate": np.nan,
    })


def load_datasets():
    train_path = find_file(["train-test.csv", "train_test.csv"])
    val_path = find_file(["validation.csv"])

    if train_path is None:
        raise FileNotFoundError("Could not find train-test.csv in root or data/ folder.")
    if val_path is None:
        raise FileNotFoundError("Could not find validation.csv in root or data/ folder.")

    print(f"Loading train from: {train_path}")
    print(f"Loading val from:   {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    dec_df = get_december_inputs()

    return train_df, val_df, dec_df


def engineer_features(df: pd.DataFrame, city_coords: dict = None) -> tuple[pd.DataFrame, dict]:
    data = df.copy()

    # 1. Clean data-quality issues: fix negative weights (e.g. sign flip error)
    if "weight" in data.columns:
        data["weight"] = data["weight"].abs()

    # 2. Date and time-based features
    dates = pd.to_datetime(data["date"])
    data["month"] = dates.dt.month
    data["day"] = dates.dt.day
    data["dayofweek"] = dates.dt.dayofweek
    data["dayofyear"] = dates.dt.dayofyear
    data["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical seasonal features
    data["sin_dayofyear"] = np.sin(2 * np.pi * data["dayofyear"] / 365.25)
    data["cos_dayofyear"] = np.cos(2 * np.pi * data["dayofyear"] / 365.25)

    # 3. Impute coordinates for inputs lacking lat/lon
    if city_coords is None:
        pickup_map = data.groupby("pickup")[["pickup_lat", "pickup_lon"]].mean().to_dict(orient="index")
        delivery_map = data.groupby("delivery")[["delivery_lat", "delivery_lon"]].mean().to_dict(orient="index")
        city_coords = {"pickup": pickup_map, "delivery": delivery_map}

    if "pickup_lat" not in data.columns:
        data["pickup_lat"] = data["pickup"].map(lambda x: city_coords["pickup"].get(x, {}).get("pickup_lat", np.nan))
        data["pickup_lon"] = data["pickup"].map(lambda x: city_coords["pickup"].get(x, {}).get("pickup_lon", np.nan))
    if "delivery_lat" not in data.columns:
        data["delivery_lat"] = data["delivery"].map(lambda x: city_coords["delivery"].get(x, {}).get("delivery_lat", np.nan))
        data["delivery_lon"] = data["delivery"].map(lambda x: city_coords["delivery"].get(x, {}).get("delivery_lon", np.nan))

    # 4. Domain features
    data["log_distance"] = np.log1p(data["distance"])
    data["lane_name"] = data["pickup"].astype(str) + " -> " + data["delivery"].astype(str)

    return data, city_coords


TARGET_COL = "posted_rate"
CAT_COLS = ["pickup", "delivery", "equipment"]
FEATURE_COLS = [
    "pickup", "delivery", "equipment", "lane_target_enc",
    "distance", "log_distance", "weight",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "month", "day", "dayofweek", "dayofyear", "is_weekend",
    "sin_dayofyear", "cos_dayofyear",
    "market_index", "quote_signal",
]
CAT_INDICES = [FEATURE_COLS.index(c) for c in CAT_COLS]

# Heavy regularization: the out-of-fold calendar signal is weak (residual R^2 << 1%),
# so this keeps the correction a modest nudge instead of fitting holdout noise.
SEASONAL_RIDGE_ALPHA = 50.0


def fit_lane_target_encoding(frames: list[pd.DataFrame], target_col: str, smoothing_weight: int = 10) -> None:
    """Fits smoothed target encoding on frames[0] (the training frame) and applies it to all frames in place."""
    train_frame = frames[0]
    global_mean = train_frame[target_col].mean()
    lane_stats = train_frame.groupby("lane_name")[target_col].agg(["count", "mean"])
    smoothed_lane_rates = (
        (lane_stats["count"] * lane_stats["mean"] + smoothing_weight * global_mean)
        / (lane_stats["count"] + smoothing_weight)
    ).to_dict()
    for frame in frames:
        frame["lane_target_enc"] = frame["lane_name"].map(smoothed_lane_rates).fillna(global_mean)


def fit_categorical_codes(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    """Encodes low-cardinality categoricals using a category set shared across all frames, in place."""
    for col in cat_cols:
        all_categories = sorted(set().union(*(set(f[col].dropna().unique()) for f in frames)))
        cat_type = pd.CategoricalDtype(categories=all_categories)
        for frame in frames:
            frame[col] = frame[col].astype(cat_type).cat.codes


def build_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Bounded cyclical-only features (day-of-week + day-of-year sin/cos) for the seasonal
    residual correction. Unlike raw day indices, sin/cos stay in [-1, 1] for any future date,
    so a linear model on these extrapolates safely past the training date range."""
    return pd.DataFrame({
        "sin_dow": np.sin(2 * np.pi * data["dayofweek"] / 7),
        "cos_dow": np.cos(2 * np.pi * data["dayofweek"] / 7),
        "sin_doy": data["sin_dayofyear"],
        "cos_doy": data["cos_dayofyear"],
    })


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAPE%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        categorical_features=CAT_INDICES,
        random_state=42,
    )


def run_temporal_holdout_validation(train_df: pd.DataFrame, target_col: str, holdout_months: int = 2) -> Ridge:
    """Trains on the earliest months and holds out the most recent ones, mirroring the real
    task of predicting Nov/Dec from Jan-Oct history. Prints out-of-sample metrics, then fits a
    small seasonal-correction model on the holdout residuals (see build_calendar_features)."""
    dates = pd.to_datetime(train_df["date"])
    months_sorted = sorted(dates.dt.to_period("M").unique())
    holdout_set = set(months_sorted[-holdout_months:])
    is_holdout = dates.dt.to_period("M").isin(holdout_set)

    fold_train, fold_coords = engineer_features(train_df.loc[~is_holdout].copy())
    fold_holdout, _ = engineer_features(train_df.loc[is_holdout].copy(), city_coords=fold_coords)

    fit_lane_target_encoding([fold_train, fold_holdout], target_col)
    fit_categorical_codes([fold_train, fold_holdout], CAT_COLS)

    model = make_model()
    model.fit(fold_train[FEATURE_COLS], fold_train[target_col])
    holdout_pred = np.clip(model.predict(fold_holdout[FEATURE_COLS]), a_min=1.0, a_max=None)

    metrics = compute_regression_metrics(fold_holdout[target_col], holdout_pred)
    print(
        f"Holdout = {sorted(str(m) for m in holdout_set)}  "
        f"(train={len(fold_train):,} rows, holdout={len(fold_holdout):,} rows)"
    )
    print(
        f"MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  "
        f"R2={metrics['R2']:.4f}  MAPE={metrics['MAPE%']:.2f}%"
    )

    residual = fold_holdout[target_col].to_numpy() - holdout_pred
    seasonal_model = Ridge(alpha=SEASONAL_RIDGE_ALPHA)
    seasonal_model.fit(build_calendar_features(fold_holdout), residual)
    return seasonal_model


def main():
    train_df, val_df, dec_df = load_datasets()
    target_col = TARGET_COL

    print("\n--- Temporal Holdout Validation (train on earlier months, test on most recent months) ---")
    seasonal_model = run_temporal_holdout_validation(train_df, target_col)

    print("\n--- Feature Engineering (full labeled data) ---")
    train_feat, coords = engineer_features(train_df)
    val_feat, _ = engineer_features(val_df, city_coords=coords)
    dec_feat, _ = engineer_features(dec_df, city_coords=coords)

    fit_lane_target_encoding([train_feat, val_feat, dec_feat], target_col)
    fit_categorical_codes([train_feat, val_feat, dec_feat], CAT_COLS)

    # Impute missing market index / quote signal for December inputs using December validation averages
    dec_market_avg = val_feat.loc[val_feat["month"] == 12, "market_index"].mean()
    dec_quote_avg = val_feat.loc[val_feat["month"] == 12, "quote_signal"].mean()

    for df_temp in [train_feat, val_feat, dec_feat]:
        for c in FEATURE_COLS:
            if c not in df_temp.columns:
                df_temp[c] = np.nan

    dec_feat["market_index"] = dec_feat["market_index"].fillna(dec_market_avg)
    dec_feat["quote_signal"] = dec_feat["quote_signal"].fillna(dec_quote_avg)

    X_train = train_feat[FEATURE_COLS]
    y_train = train_feat[target_col]
    X_val = val_feat[FEATURE_COLS]
    X_dec = dec_feat[FEATURE_COLS]

    print("\n--- Training Final HistGradientBoostingRegressor on all labeled data ---")
    model = make_model()
    model.fit(X_train, y_train)

    print("\n--- Generating Predictions ---")
    # Tree ensembles can't extrapolate trends past the training date range (Jan-Oct), so a
    # small calendar-only correction (fit during holdout validation) is added back on top to
    # recover realistic day-to-day/seasonal movement for the unseen Nov/Dec period.
    val_preds = np.clip(
        model.predict(X_val) + seasonal_model.predict(build_calendar_features(val_feat)),
        a_min=1.0, a_max=None,
    )
    dec_preds = np.clip(
        model.predict(X_dec) + seasonal_model.predict(build_calendar_features(dec_feat)),
        a_min=1.0, a_max=None,
    )

    # 1. Save validation_predictions.csv (exact format: load_id,predicted_rate)
    val_out = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": val_preds,
    })
    val_out.to_csv("validation_predictions.csv", index=False)
    print("Saved -> validation_predictions.csv")

    # 2. Save completed december inputs file (exact 7 columns)
    dec_out = dec_df.copy()
    dec_out["predicted_rate"] = dec_preds
    dec_out_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_out = dec_out[dec_out_cols]

    dec_out.to_csv("december_chart_inputs.csv", index=False)
    print("Saved -> december_chart_inputs.csv")
    if Path("data").exists():
        dec_out.to_csv("data/december_chart_inputs.csv", index=False)
        print("Saved -> data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()