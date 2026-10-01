from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COLUMN = "posted_rate"
CATEGORY_COLUMNS = ["pickup", "delivery", "equipment"]
FEATURE_COLUMNS = [
    "pickup", "delivery", "equipment", "lane_avg_rate",
    "distance", "log_distance", "weight",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "month", "day", "dayofweek", "dayofyear", "is_weekend",
    "sin_dayofyear", "cos_dayofyear",
    "market_index", "quote_signal",
]
CATEGORY_COLUMN_POSITIONS = [FEATURE_COLUMNS.index(column) for column in CATEGORY_COLUMNS]


def find_file(candidates: list[str]) -> Path | None:
    for name in candidates:
        path = Path(name)
        if path.exists():
            return path
        path_in_data = Path("data") / name
        if path_in_data.exists():
            return path_in_data
    return None


def get_december_inputs() -> pd.DataFrame:
    dec_path = find_file(["december_chart_inputs.csv", "december-chart-inputs.csv"])
    if dec_path is not None:
        print(f"Loading December scenario from: {dec_path}")
        return pd.read_csv(dec_path)

    print("December scenario file not found. Generating it from the assessment spec...")
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
        raise FileNotFoundError("Could not find train-test.csv in the data/ folder.")
    if val_path is None:
        raise FileNotFoundError("Could not find validation.csv in the data/ folder.")

    print(f"Loading train from: {train_path}")
    print(f"Loading val from:   {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    dec_df = get_december_inputs()

    return train_df, val_df, dec_df


def add_date_features(data: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(data["date"])
    data["month"] = dates.dt.month
    data["day"] = dates.dt.day
    data["dayofweek"] = dates.dt.dayofweek
    data["dayofyear"] = dates.dt.dayofyear
    data["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)

    data["sin_dayofyear"] = np.sin(2 * np.pi * data["dayofyear"] / 365.25)
    data["cos_dayofyear"] = np.cos(2 * np.pi * data["dayofyear"] / 365.25)
    return data


def get_city_coordinates(data: pd.DataFrame) -> dict:
    pickup_coords = data.groupby("pickup")[["pickup_lat", "pickup_lon"]].mean().to_dict(orient="index")
    delivery_coords = data.groupby("delivery")[["delivery_lat", "delivery_lon"]].mean().to_dict(orient="index")
    return {"pickup": pickup_coords, "delivery": delivery_coords}


def fill_missing_coordinates(data: pd.DataFrame, city_coords: dict) -> pd.DataFrame:
    if "pickup_lat" not in data.columns:
        data["pickup_lat"] = data["pickup"].map(lambda city: city_coords["pickup"].get(city, {}).get("pickup_lat", np.nan))
        data["pickup_lon"] = data["pickup"].map(lambda city: city_coords["pickup"].get(city, {}).get("pickup_lon", np.nan))
    if "delivery_lat" not in data.columns:
        data["delivery_lat"] = data["delivery"].map(lambda city: city_coords["delivery"].get(city, {}).get("delivery_lat", np.nan))
        data["delivery_lon"] = data["delivery"].map(lambda city: city_coords["delivery"].get(city, {}).get("delivery_lon", np.nan))
    return data


def engineer_features(df: pd.DataFrame, city_coords: dict = None) -> tuple[pd.DataFrame, dict]:
    data = df.copy()

    if "weight" in data.columns:
        data["weight"] = data["weight"].abs()

    data = add_date_features(data)

    if city_coords is None:
        city_coords = get_city_coordinates(data)
    data = fill_missing_coordinates(data, city_coords)

    data["log_distance"] = np.log1p(data["distance"])
    data["lane_name"] = data["pickup"].astype(str) + " -> " + data["delivery"].astype(str)

    return data, city_coords


def get_lane_average_rates(train_df: pd.DataFrame, min_loads_for_full_trust: int = 10) -> tuple[dict, float]:
    overall_average = train_df[TARGET_COLUMN].mean()
    lane_stats = train_df.groupby("lane_name")[TARGET_COLUMN].agg(["count", "mean"])
    blended_average = (
        lane_stats["count"] * lane_stats["mean"] + min_loads_for_full_trust * overall_average
    ) / (lane_stats["count"] + min_loads_for_full_trust)
    return blended_average.to_dict(), overall_average


def get_category_type(dataframes: list[pd.DataFrame], column: str) -> pd.CategoricalDtype:
    values = set()
    for df in dataframes:
        values.update(df[column].dropna().unique())
    return pd.CategoricalDtype(categories=sorted(values))


def fill_missing_column(df: pd.DataFrame, column: str, fallback_value: float) -> None:
    if column not in df.columns:
        df[column] = np.nan
    df[column] = df[column].fillna(fallback_value)


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
        categorical_features=CATEGORY_COLUMN_POSITIONS,
        random_state=42,
    )


def add_model_columns(train_df: pd.DataFrame, other_dfs: list[pd.DataFrame]) -> None:
    all_dfs = [train_df] + other_dfs
    lane_rates, overall_average = get_lane_average_rates(train_df)
    for df in all_dfs:
        df["lane_avg_rate"] = df["lane_name"].map(lane_rates).fillna(overall_average)

    for column in CATEGORY_COLUMNS:
        category_type = get_category_type(all_dfs, column)
        for df in all_dfs:
            df[column] = df[column].astype(category_type).cat.codes


def validate_with_holdout(train_df: pd.DataFrame, holdout_months: int = 2) -> None:
    dates = pd.to_datetime(train_df["date"])
    all_months = sorted(dates.dt.to_period("M").unique())
    holdout_months_list = all_months[-holdout_months:]
    is_holdout = dates.dt.to_period("M").isin(holdout_months_list)

    fold_train, fold_coords = engineer_features(train_df.loc[~is_holdout].copy())
    fold_holdout, _ = engineer_features(train_df.loc[is_holdout].copy(), city_coords=fold_coords)
    add_model_columns(fold_train, [fold_holdout])

    model = make_model()
    model.fit(fold_train[FEATURE_COLUMNS], fold_train[TARGET_COLUMN])
    holdout_predictions = np.clip(model.predict(fold_holdout[FEATURE_COLUMNS]), a_min=1.0, a_max=None)

    metrics = compute_regression_metrics(fold_holdout[TARGET_COLUMN], holdout_predictions)
    print(f"Holdout months: {[str(month) for month in holdout_months_list]}")
    print(f"Trained on {len(fold_train):,} rows, tested on {len(fold_holdout):,} rows")
    print(
        f"MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  "
        f"R2={metrics['R2']:.4f}  MAPE={metrics['MAPE%']:.2f}%"
    )


def get_dayofweek_adjustment(train_df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(train_df["date"])
    rates_by_day = train_df.groupby(dates.dt.dayofweek)[TARGET_COLUMN].mean()
    overall_average = train_df[TARGET_COLUMN].mean()
    return (rates_by_day - overall_average).to_dict()


def apply_dayofweek_adjustment(predictions: np.ndarray, df: pd.DataFrame, adjustment: dict) -> np.ndarray:
    return predictions + df["dayofweek"].map(adjustment).fillna(0.0).to_numpy()


def main():
    train_df, val_df, dec_df = load_datasets()

    print("\n--- Validating the model on data it hasn't seen (holdout) ---")
    validate_with_holdout(train_df)

    print("\n--- Checking for day-of-week patterns ---")
    dayofweek_adjustment = get_dayofweek_adjustment(train_df)
    print({day: round(amount, 2) for day, amount in dayofweek_adjustment.items()})

    print("\n--- Preparing features on the full dataset ---")
    train_feat, coords = engineer_features(train_df)
    val_feat, _ = engineer_features(val_df, city_coords=coords)
    dec_feat, _ = engineer_features(dec_df, city_coords=coords)
    add_model_columns(train_feat, [val_feat, dec_feat])

    dec_market_avg = val_feat.loc[val_feat["month"] == 12, "market_index"].mean()
    dec_quote_avg = val_feat.loc[val_feat["month"] == 12, "quote_signal"].mean()
    fill_missing_column(dec_feat, "market_index", dec_market_avg)
    fill_missing_column(dec_feat, "quote_signal", dec_quote_avg)

    print("\n--- Training the final model on all labeled data ---")
    model = make_model()
    model.fit(train_feat[FEATURE_COLUMNS], train_feat[TARGET_COLUMN])

    print("\n--- Generating predictions ---")
    val_preds = apply_dayofweek_adjustment(model.predict(val_feat[FEATURE_COLUMNS]), val_feat, dayofweek_adjustment)
    dec_preds = apply_dayofweek_adjustment(model.predict(dec_feat[FEATURE_COLUMNS]), dec_feat, dayofweek_adjustment)
    val_preds = np.clip(val_preds, a_min=1.0, a_max=None)
    dec_preds = np.clip(dec_preds, a_min=1.0, a_max=None)

    val_out = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": val_preds,
    })
    val_out.to_csv("validation_predictions.csv", index=False)
    print("Saved -> validation_predictions.csv")

    dec_out = dec_df.copy()
    dec_out["predicted_rate"] = dec_preds
    dec_out = dec_out[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]]
    dec_out.to_csv("data/december_chart_inputs.csv", index=False)
    print("Saved -> data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
