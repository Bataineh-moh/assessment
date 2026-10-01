import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


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
    train_path = find_file(["C:\\Users\\DELL\\Desktop\\assessment\\data\\train-test.csv", "train_test.csv"])
    val_path = find_file(["C:\\Users\\DELL\\Desktop\\assessment\\data\\validation.csv"])

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


def main():
    train_df, val_df, dec_df = load_datasets()

    print("\n--- Feature Engineering ---")
    train_feat, coords = engineer_features(train_df)
    val_feat, _ = engineer_features(val_df, city_coords=coords)
    dec_feat, _ = engineer_features(dec_df, city_coords=coords)

    target_col = "posted_rate"

    # Smoothed Target Encoding for the high-cardinality lane feature
    global_mean = train_feat[target_col].mean()
    lane_stats = train_feat.groupby("lane_name")[target_col].agg(["count", "mean"])
    smoothing_weight = 10
    smoothed_lane_rates = (
        (lane_stats["count"] * lane_stats["mean"] + smoothing_weight * global_mean)
        / (lane_stats["count"] + smoothing_weight)
    ).to_dict()

    train_feat["lane_target_enc"] = train_feat["lane_name"].map(smoothed_lane_rates).fillna(global_mean)
    val_feat["lane_target_enc"] = val_feat["lane_name"].map(smoothed_lane_rates).fillna(global_mean)
    dec_feat["lane_target_enc"] = dec_feat["lane_name"].map(smoothed_lane_rates).fillna(global_mean)

    # Encode only low-cardinality categoricals: pickup (64), delivery (64), equipment (3)
    cat_cols = ["pickup", "delivery", "equipment"]
    for col in cat_cols:
        all_categories = sorted(list(
            set(train_feat[col].dropna().unique())
            | set(val_feat[col].dropna().unique())
            | set(dec_feat[col].dropna().unique())
        ))
        cat_type = pd.CategoricalDtype(categories=all_categories)
        train_feat[col] = train_feat[col].astype(cat_type).cat.codes
        val_feat[col] = val_feat[col].astype(cat_type).cat.codes
        dec_feat[col] = dec_feat[col].astype(cat_type).cat.codes

    # Numeric & engineered features
    feature_cols = [
        "pickup", "delivery", "equipment", "lane_target_enc",
        "distance", "log_distance", "weight",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "month", "day", "dayofweek", "dayofyear", "is_weekend",
        "sin_dayofyear", "cos_dayofyear",
        "market_index", "quote_signal",
    ]

    # Impute missing market index / quote signal for December inputs using December validation averages
    dec_market_avg = val_feat.loc[val_feat["month"] == 12, "market_index"].mean()
    dec_quote_avg = val_feat.loc[val_feat["month"] == 12, "quote_signal"].mean()

    for df_temp in [train_feat, val_feat, dec_feat]:
        for c in feature_cols:
            if c not in df_temp.columns:
                df_temp[c] = np.nan

    dec_feat["market_index"] = dec_feat["market_index"].fillna(dec_market_avg)
    dec_feat["quote_signal"] = dec_feat["quote_signal"].fillna(dec_quote_avg)

    X_train = train_feat[feature_cols]
    y_train = train_feat[target_col]
    X_val = val_feat[feature_cols]
    X_dec = dec_feat[feature_cols]

    # Only pass low-cardinality categorical indices (0, 1, 2)
    cat_indices = [feature_cols.index(c) for c in cat_cols]

    print("\n--- Training HistGradientBoostingRegressor ---")
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        categorical_features=cat_indices,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\n--- Generating Predictions ---")
    val_preds = np.clip(model.predict(X_val), a_min=1.0, a_max=None)
    dec_preds = np.clip(model.predict(X_dec), a_min=1.0, a_max=None)

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