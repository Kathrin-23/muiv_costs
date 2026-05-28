import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = [
    "year",
    "region",
    "education_level",
    "program_name",
    "study_format",
    "duration_months",
    "base_price",
    "competitor_price",
    "demand_index",
    "salary_index",
    "advertising_spend",
    "discount_percent",
    "student_count",
    "final_price",
]

FEATURE_COLUMNS = [
    "year",
    "region",
    "education_level",
    "program_name",
    "study_format",
    "duration_months",
    "base_price",
    "competitor_price",
    "demand_index",
    "salary_index",
    "advertising_spend",
    "discount_percent",
    "student_count",
]

TARGET_COLUMN = "final_price"
CATEGORICAL_COLUMNS = ["region", "education_level", "program_name", "study_format"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

COLUMN_ALIASES = {
    "Год": "year",
    "Регион": "region",
    "Уровень образования": "education_level",
    "Название услуги": "program_name",
    "Направление подготовки": "program_name",
    "Форма обучения": "study_format",
    "Длительность обучения": "duration_months",
    "Текущая цена": "base_price",
    "Средняя цена конкурентов": "competitor_price",
    "Количество заявок": "student_count",
    "Скидка": "discount_percent",
    "Прогнозная цена": "final_price",
    "Индекс спроса": "demand_index",
    "Индекс зарплат": "salary_index",
    "Расходы на продвижение": "advertising_spend",
}


def read_dataset(path):
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Файл набора данных не найден: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return normalize_dataset(df)


def normalize_dataset(df):
    normalized = df.copy()
    normalized.columns = [str(column).replace("\ufeff", "").strip() for column in normalized.columns]
    rename_map = {column: COLUMN_ALIASES.get(column, column) for column in normalized.columns}
    normalized = normalized.rename(columns=rename_map)
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]
    if "base_price" not in normalized.columns and "final_price" in normalized.columns:
        normalized["base_price"] = normalized["final_price"]
    if "final_price" not in normalized.columns and "base_price" in normalized.columns:
        normalized["final_price"] = normalized["base_price"]
    if "demand_index" not in normalized.columns and "student_count" in normalized.columns:
        numeric_students = pd.to_numeric(normalized["student_count"], errors="coerce")
        min_students = numeric_students.min()
        max_students = numeric_students.max()
        if pd.notna(min_students) and pd.notna(max_students) and max_students != min_students:
            normalized["demand_index"] = 40 + (numeric_students - min_students) / (max_students - min_students) * 60
        else:
            normalized["demand_index"] = 65
    if "salary_index" not in normalized.columns:
        normalized["salary_index"] = 70
    if "advertising_spend" not in normalized.columns:
        normalized["advertising_spend"] = 0
    for column in REQUIRED_COLUMNS:
        if column not in normalized.columns:
            raise ValueError(f"В наборе данных отсутствует обязательный столбец: {column}")
    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        normalized[column] = normalized[column].fillna("не указано").astype(str).str.strip()
    normalized = normalized.dropna(subset=NUMERIC_COLUMNS + [TARGET_COLUMN])
    normalized = normalized[REQUIRED_COLUMNS]
    if normalized.empty:
        raise ValueError("После очистки набор данных не содержит пригодных записей")
    return normalized


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def build_model(model_name="random_forest"):
    preprocessor = build_preprocessor()
    if model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=42, n_estimators=160, max_depth=3)
    elif model_name == "ridge":
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])


def train_model(dataset_path, model_path, metrics_path=None, model_name="random_forest"):
    df = read_dataset(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
    model = build_model(model_name=model_name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    metrics.update({
        "model_name": model_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(df)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    if metrics_path:
        Path(metrics_path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Файл обученной модели не найден")
    return joblib.load(model_path)


def calculate_metrics(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def validate_input(payload):
    errors = []
    cleaned = {}
    for column in FEATURE_COLUMNS:
        value = payload.get(column)
        if value in (None, ""):
            errors.append(f"Не заполнено поле {column}")
            continue
        if column in NUMERIC_COLUMNS:
            try:
                cleaned[column] = float(value)
            except ValueError:
                errors.append(f"Поле {column} должно быть числом")
        else:
            cleaned[column] = str(value).strip()
    if errors:
        raise ValueError("; ".join(errors))
    cleaned["year"] = int(cleaned["year"])
    cleaned["duration_months"] = int(cleaned["duration_months"])
    cleaned["student_count"] = int(cleaned["student_count"])
    return cleaned


def predict_price(model_path, payload):
    cleaned = validate_input(payload)
    model = load_model(model_path)
    input_frame = pd.DataFrame([cleaned], columns=FEATURE_COLUMNS)
    predicted_price = float(model.predict(input_frame)[0])
    recommendation = build_price_recommendation(predicted_price, cleaned)
    return {
        "predicted_price": round(predicted_price, 2),
        "recommendation": recommendation,
        "input": cleaned,
    }


def build_price_recommendation(predicted_price, payload):
    competitor_price = float(payload.get("competitor_price", predicted_price))
    base_price = float(payload.get("base_price", predicted_price))
    demand_index = float(payload.get("demand_index", 50))
    discount = float(payload.get("discount_percent", 0))
    ratio_to_competitor = predicted_price / competitor_price if competitor_price else 1
    ratio_to_base = predicted_price / base_price if base_price else 1

    if ratio_to_competitor > 1.12 and demand_index < 55:
        return "Рекомендуется ограничить рост цены и усилить скидочные предложения, так как прогноз выше рынка при умеренном спросе."
    if ratio_to_competitor < 0.92 and demand_index > 70:
        return "Есть резерв для повышения цены: прогноз ниже уровня конкурентов при повышенном спросе."
    if discount > 15 and ratio_to_base < 1.0:
        return "Скидочная политика заметно снижает итоговую цену; требуется проверить экономическую целесообразность акции."
    if demand_index > 80:
        return "Спрос высокий, поэтому допустимо аккуратное повышение цены при сохранении конкурентного позиционирования."
    return "Прогнозная цена находится в допустимом диапазоне; рекомендуется использовать ее как базовый ориентир для планирования."


def dataset_summary(dataset_path):
    df = read_dataset(dataset_path)
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "mean_price": round(float(df["final_price"].mean()), 2),
        "median_price": round(float(df["final_price"].median()), 2),
        "min_price": round(float(df["final_price"].min()), 2),
        "max_price": round(float(df["final_price"].max()), 2),
        "mean_competitor_price": round(float(df["competitor_price"].mean()), 2),
        "mean_discount": round(float(df["discount_percent"].mean()), 2),
        "total_students": int(df["student_count"].sum()),
        "regions": sorted(df["region"].unique().tolist()),
        "education_levels": sorted(df["education_level"].unique().tolist()),
        "study_formats": sorted(df["study_format"].unique().tolist()),
    }
    return summary


def price_dynamics(dataset_path):
    df = read_dataset(dataset_path)
    grouped = df.groupby("year", as_index=False).agg(
        mean_price=("final_price", "mean"),
        mean_demand=("demand_index", "mean"),
        mean_competitor_price=("competitor_price", "mean"),
        observations=("final_price", "count"),
    )
    grouped["mean_price"] = grouped["mean_price"].round(2)
    grouped["mean_demand"] = grouped["mean_demand"].round(2)
    grouped["mean_competitor_price"] = grouped["mean_competitor_price"].round(2)
    grouped["growth_percent"] = grouped["mean_price"].pct_change().fillna(0).mul(100).round(2)
    return grouped.to_dict(orient="records")


def program_rating(dataset_path, limit=10):
    df = read_dataset(dataset_path)
    grouped = df.groupby("program_name", as_index=False).agg(
        mean_price=("final_price", "mean"),
        mean_demand=("demand_index", "mean"),
        students=("student_count", "sum"),
    )
    grouped["mean_price"] = grouped["mean_price"].round(2)
    grouped["mean_demand"] = grouped["mean_demand"].round(2)
    grouped = grouped.sort_values(["mean_demand", "students"], ascending=False).head(limit)
    return grouped.to_dict(orient="records")


def category_analysis(dataset_path, column):
    df = read_dataset(dataset_path)
    grouped = df.groupby(column, as_index=False).agg(
        mean_price=("final_price", "mean"),
        min_price=("final_price", "min"),
        max_price=("final_price", "max"),
        mean_competitor_price=("competitor_price", "mean"),
        mean_demand=("demand_index", "mean"),
        observations=("final_price", "count"),
    )
    for price_column in ["mean_price", "min_price", "max_price", "mean_competitor_price", "mean_demand"]:
        grouped[price_column] = grouped[price_column].round(2)
    grouped["deviation_from_competitors"] = (grouped["mean_price"] - grouped["mean_competitor_price"]).round(2)
    grouped = grouped.sort_values("mean_price", ascending=False)
    return grouped.to_dict(orient="records")


def competitor_analysis(dataset_path):
    df = read_dataset(dataset_path)
    grouped = df.groupby("year", as_index=False).agg(
        mean_price=("final_price", "mean"),
        mean_competitor_price=("competitor_price", "mean"),
        observations=("final_price", "count"),
    )
    grouped["mean_price"] = grouped["mean_price"].round(2)
    grouped["mean_competitor_price"] = grouped["mean_competitor_price"].round(2)
    grouped["difference"] = (grouped["mean_price"] - grouped["mean_competitor_price"]).round(2)
    grouped["difference_percent"] = (grouped["difference"] / grouped["mean_competitor_price"] * 100).round(2)
    return grouped.to_dict(orient="records")


def demand_analysis(dataset_path):
    df = read_dataset(dataset_path)
    grouped = df.groupby("program_name", as_index=False).agg(
        mean_price=("final_price", "mean"),
        applications=("student_count", "sum"),
        mean_demand=("demand_index", "mean"),
    )
    grouped["mean_price"] = grouped["mean_price"].round(2)
    grouped["mean_demand"] = grouped["mean_demand"].round(2)
    grouped = grouped.sort_values("applications", ascending=False).head(12)
    return grouped.to_dict(orient="records")


def dataset_preview(dataset_path, limit=15):
    df = read_dataset(dataset_path).head(limit)
    return df.to_dict(orient="records"), list(df.columns)


def cross_validate_model(dataset_path, model_name="random_forest"):
    df = read_dataset(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    model = build_model(model_name=model_name)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    positive_scores = np.abs(scores)
    return {
        "folds": 5,
        "mae_mean": round(float(positive_scores.mean()), 2),
        "mae_std": round(float(positive_scores.std()), 2),
    }


def prepare_forecast_payload_from_service(service, year=None):
    latest_price = service.last_price()
    year = year or datetime.now().year + 1
    latest_demand = sorted(service.demand_indicators, key=lambda item: item.year, reverse=True)[0] if service.demand_indicators else None
    latest_competitor = sorted(service.competitor_prices, key=lambda item: item.year, reverse=True)[0] if service.competitor_prices else None
    return {
        "year": year,
        "region": service.region,
        "education_level": service.program.education_level,
        "program_name": service.program.name,
        "study_format": service.study_format,
        "duration_months": service.program.duration_months,
        "base_price": latest_price,
        "competitor_price": latest_competitor.price if latest_competitor else latest_price * 0.98,
        "demand_index": latest_demand.demand_index if latest_demand else 65,
        "salary_index": latest_demand.salary_index if latest_demand else 70,
        "advertising_spend": latest_demand.advertising_spend if latest_demand else 120000,
        "discount_percent": 7,
        "student_count": 60,
    }
