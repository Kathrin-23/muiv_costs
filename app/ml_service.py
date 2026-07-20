"""Подготовка данных, обучение моделей и прогнозирование стоимости.

Автор: Горячевская Екатерина Николевна
Тема ВКР: «Анализ и прогнозирование ценообразования на образовательные услуги».

Модуль содержит единый воспроизводимый ML-конвейер: очистку данных, разбиение
на три выборки, подбор гиперпараметров, сравнение алгоритмов, диагностику и
сохранение каждой обученной модели в отдельный файл.
"""

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = [
    "year",
    "organization",
    "program_name",
    "base_price",
    "competitor_price",
    "student_count",
    "admission_score",
    "salary_index",
    "final_price",
]

FEATURE_COLUMNS = [
    "year",
    "organization",
    "program_name",
    "base_price",
    "competitor_price",
    "student_count",
    "admission_score",
    "salary_index",
]

TARGET_COLUMN = "final_price"
CATEGORICAL_COLUMNS = ["organization", "program_name"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

MODEL_FILENAMES = {
    "random_forest": "model_random_forest.pkl",
    "gradient_boosting": "model_gradient_boosting.pkl",
    "ridge": "model_ridge.pkl",
}
MODEL_DISPLAY_NAMES = {
    "random_forest": "Случайный лес",
    "gradient_boosting": "Градиентный бустинг",
    "ridge": "Гребневая регрессия Ridge",
}
MODEL_PARAMETER_GRIDS = {
    "random_forest": {
        "regressor__n_estimators": [80, 120],
        "regressor__max_depth": [10, 16],
        "regressor__min_samples_leaf": [1, 3],
    },
    "gradient_boosting": {
        "regressor__n_estimators": [100, 180],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [2, 3],
    },
    "ridge": {
        "regressor__alpha": [0.1, 1.0, 10.0, 100.0],
    },
}

COLUMN_ALIASES = {
    "Год": "year",
    "Вуз": "organization",
    "Образовательная организация": "organization",
    "Название услуги": "program_name",
    "Направление подготовки": "program_name",
    "Текущая цена": "base_price",
    "Средняя цена конкурентов": "competitor_price",
    "Зачислено на платные места": "student_count",
    "Средний балл приема": "admission_score",
    "Прогнозная цена": "final_price",
    "Индекс зарплат": "salary_index",
}


def read_dataset(path):
    """Прочитать CSV/XLSX и привести таблицу к схеме ML-модуля."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Файл набора данных не найден: {dataset_path}")
    if dataset_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    return normalize_dataset(df)


def normalize_dataset(df):
    """Очистить названия, типы и пропуски, затем проверить обязательные поля."""

    normalized = df.copy()
    normalized.columns = [str(column).replace("\ufeff", "").strip() for column in normalized.columns]
    rename_map = {column: COLUMN_ALIASES.get(column, column) for column in normalized.columns}
    normalized = normalized.rename(columns=rename_map)
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]
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
    """Создать общий препроцессор категориальных и числовых признаков."""

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def build_model(model_name="random_forest"):
    """Создать конвейер предобработки и выбранного регрессора."""

    if model_name not in MODEL_FILENAMES:
        raise ValueError(f"Неизвестная модель: {model_name}")
    preprocessor = build_preprocessor()
    if model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=42)
    elif model_name == "ridge":
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])


def split_dataset(df, random_state=42):
    """Разделить данные строго по времени: прошлые годы / предпоследний / последний."""

    del random_state  # параметр оставлен для совместимости с прежним API
    ordered = df.sort_values("year").reset_index(drop=True)
    years = sorted(int(year) for year in ordered["year"].unique())
    if len(years) < 3:
        raise ValueError("Для временного разбиения требуется минимум три разных года")
    validation_year = years[-2]
    test_year = years[-1]
    train_frame = ordered[ordered["year"] < validation_year]
    validation_frame = ordered[ordered["year"] == validation_year]
    test_frame = ordered[ordered["year"] == test_year]
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Не удалось сформировать непустые временные выборки")
    return {
        "train": (train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN]),
        "validation": (
            validation_frame[FEATURE_COLUMNS],
            validation_frame[TARGET_COLUMN],
        ),
        "test": (test_frame[FEATURE_COLUMNS], test_frame[TARGET_COLUMN]),
    }


def _clean_best_params(params):
    """Убрать служебный префикс конвейера из названий параметров."""

    return {
        key.replace("regressor__", ""): value
        for key, value in params.items()
    }


def _calculate_split_metrics(model, splits):
    """Рассчитать одинаковый набор метрик на трех выборках."""

    return {
        split_name: calculate_metrics(y_values, model.predict(X_values))
        for split_name, (X_values, y_values) in splits.items()
    }


def _cross_validation_metrics(model, X_train, y_train):
    """Оценить настроенную модель пятикратной кросс-валидацией."""

    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    return {
        "folds": 5,
        "mae_mean": round(float(-scores["test_mae"].mean()), 2),
        "mae_std": round(float(scores["test_mae"].std()), 2),
        "rmse_mean": round(float(-scores["test_rmse"].mean()), 2),
        "rmse_std": round(float(scores["test_rmse"].std()), 2),
        "r2_mean": round(float(scores["test_r2"].mean()), 4),
        "r2_std": round(float(scores["test_r2"].std()), 4),
    }


def _overfitting_conclusion(metrics):
    """Сформировать интерпретируемый вывод о возможном переобучении."""

    train = metrics["train"]
    validation = metrics["validation"]
    r2_gap = train["r2"] - validation["r2"]
    rmse_ratio = (
        validation["rmse"] / train["rmse"]
        if train["rmse"] > 0
        else float("inf")
    )
    detected = r2_gap > 0.15 or rmse_ratio > 1.5
    if detected:
        conclusion = (
            "Есть признаки переобучения: качество на обучающей выборке заметно "
            "выше качества на валидационной выборке."
        )
    else:
        conclusion = (
            "Выраженных признаков переобучения не обнаружено: метрики обучающей "
            "и валидационной выборок сопоставимы."
        )
    return {
        "detected": detected,
        "r2_gap": round(float(r2_gap), 4),
        "validation_to_train_rmse": round(float(rmse_ratio), 4),
        "conclusion": conclusion,
    }


def _feature_importance(model):
    """Получить важность преобразованных признаков для дерева или Ridge."""

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return []
    items = [
        {"feature": str(name), "importance": round(float(value), 8)}
        for name, value in zip(feature_names, values)
    ]
    return sorted(items, key=lambda item: item["importance"], reverse=True)


def generate_diagnostics(model, model_name, splits, metrics, output_dir):
    """Сохранить пять графиков диагностики и JSON важности признаков."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_test, y_test = splits["test"]
    predictions = model.predict(X_test)
    residuals = np.asarray(y_test) - predictions
    files = {}

    def save_figure(key, filename):
        path = output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close()
        files[key] = path.name

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.65)
    bounds = [min(float(y_test.min()), float(predictions.min())), max(float(y_test.max()), float(predictions.max()))]
    plt.plot(bounds, bounds, "--", color="crimson")
    plt.xlabel("Фактическая цена, руб.")
    plt.ylabel("Прогнозная цена, руб.")
    plt.title(f"{MODEL_DISPLAY_NAMES[model_name]}: факт и прогноз")
    save_figure("actual_vs_predicted", f"{model_name}_actual_vs_predicted.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(predictions, residuals, alpha=0.65)
    plt.axhline(0, linestyle="--", color="crimson")
    plt.xlabel("Прогнозная цена, руб.")
    plt.ylabel("Остаток, руб.")
    plt.title(f"{MODEL_DISPLAY_NAMES[model_name]}: остатки")
    save_figure("residuals", f"{model_name}_residuals.png")

    plt.figure(figsize=(7, 5))
    plt.hist(np.abs(residuals), bins=20, edgecolor="white")
    plt.xlabel("Абсолютная ошибка, руб.")
    plt.ylabel("Количество наблюдений")
    plt.title(f"{MODEL_DISPLAY_NAMES[model_name]}: распределение ошибок")
    save_figure("error_distribution", f"{model_name}_error_distribution.png")

    split_names = ["train", "validation", "test"]
    plt.figure(figsize=(8, 5))
    positions = np.arange(len(split_names))
    width = 0.35
    plt.bar(positions - width / 2, [metrics[name]["mae"] for name in split_names], width, label="MAE")
    plt.bar(positions + width / 2, [metrics[name]["rmse"] for name in split_names], width, label="RMSE")
    plt.xticks(positions, ["Обучение", "Валидация", "Тест"])
    plt.ylabel("Ошибка, руб.")
    plt.title(f"{MODEL_DISPLAY_NAMES[model_name]}: метрики выборок")
    plt.legend()
    save_figure("split_metrics", f"{model_name}_split_metrics.png")

    importance = _feature_importance(model)
    importance_path = output_dir / f"{model_name}_feature_importance.json"
    importance_path.write_text(
        json.dumps(importance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files["feature_importance_json"] = importance_path.name
    top = importance[:15]
    if top:
        plt.figure(figsize=(9, 6))
        plt.barh(
            [item["feature"] for item in reversed(top)],
            [item["importance"] for item in reversed(top)],
        )
        plt.xlabel("Важность")
        plt.title(f"{MODEL_DISPLAY_NAMES[model_name]}: важность признаков")
        save_figure("feature_importance", f"{model_name}_feature_importance.png")
    return files


def _train_experiment(df, model_name, diagnostics_dir=None):
    """Настроить одну модель и вернуть финальный конвейер и полный отчет."""

    splits = split_dataset(df)
    X_train, y_train = splits["train"]
    started = perf_counter()
    search = GridSearchCV(
        build_model(model_name),
        MODEL_PARAMETER_GRIDS[model_name],
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    evaluation_model = search.best_estimator_
    metrics = _calculate_split_metrics(evaluation_model, splits)
    cv_metrics = _cross_validation_metrics(evaluation_model, X_train, y_train)
    diagnostics = {}
    if diagnostics_dir:
        diagnostics = generate_diagnostics(
            evaluation_model,
            model_name,
            splits,
            metrics,
            diagnostics_dir,
        )

    # После независимой оценки объединяем train и validation для финальной модели.
    X_final = pd.concat([splits["train"][0], splits["validation"][0]])
    y_final = pd.concat([splits["train"][1], splits["validation"][1]])
    final_model = clone(evaluation_model)
    final_model.fit(X_final, y_final)
    duration = perf_counter() - started
    report = {
        "model_name": model_name,
        "display_name": MODEL_DISPLAY_NAMES[model_name],
        "best_params": _clean_best_params(search.best_params_),
        "metrics": metrics,
        "cross_validation": cv_metrics,
        "overfitting": _overfitting_conclusion(metrics),
        "duration_seconds": round(float(duration), 3),
        "rows": {
            "train": int(len(splits["train"][0])),
            "validation": int(len(splits["validation"][0])),
            "test": int(len(splits["test"][0])),
            "dataset": int(len(df)),
        },
        "split_years": {
            split_name: sorted(
                int(year) for year in values[0]["year"].unique()
            )
            for split_name, values in splits.items()
        },
        "diagnostics": diagnostics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    return final_model, report


def train_all_models(dataset_path, model_dir, registry_path=None, diagnostics_dir=None):
    """Обучить три модели, сохранить артефакты и выбрать лучшую по validation RMSE."""

    df = read_dataset(dataset_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = Path(diagnostics_dir or model_dir / "diagnostics")
    reports = {}
    for model_name, filename in MODEL_FILENAMES.items():
        model, report = _train_experiment(
            df,
            model_name,
            diagnostics_dir=diagnostics_dir,
        )
        model_path = model_dir / filename
        joblib.dump(model, model_path)
        report["model_path"] = filename
        metrics_path = model_dir / f"metrics_{model_name}.json"
        metrics_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reports[model_name] = report

    selected_model = min(
        reports,
        key=lambda name: reports[name]["metrics"]["validation"]["rmse"],
    )
    registry = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_metric": "validation.rmse",
        "selected_model": selected_model,
        "models": reports,
    }
    registry_path = Path(registry_path or model_dir / "model_registry.json")
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    comparison_rows = []
    for model_name, report in reports.items():
        row = {
            "model_name": model_name,
            "display_name": report["display_name"],
            "duration_seconds": report["duration_seconds"],
            "cv_mae_mean": report["cross_validation"]["mae_mean"],
            "cv_rmse_mean": report["cross_validation"]["rmse_mean"],
            "best_params": json.dumps(report["best_params"], ensure_ascii=False),
            "selected": model_name == selected_model,
        }
        for split_name, split_metrics in report["metrics"].items():
            for metric_name, value in split_metrics.items():
                row[f"{split_name}_{metric_name}"] = value
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(
        model_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Файл сохраняет обратную совместимость со старой страницей метрик.
    legacy_metrics = {
        **reports[selected_model]["metrics"]["test"],
        "model_name": selected_model,
        "dataset_rows": reports[selected_model]["rows"]["dataset"],
        "trained_at": reports[selected_model]["trained_at"],
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(legacy_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return registry


def train_model(dataset_path, model_path, metrics_path=None, model_name="random_forest"):
    """Обучить одну настроенную модель (совместимый программный интерфейс)."""

    df = read_dataset(dataset_path)
    model, report = _train_experiment(df, model_name)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    if metrics_path:
        Path(metrics_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def load_model(model_path):
    """Загрузить сериализованный конвейер модели."""

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Файл обученной модели не найден")
    return joblib.load(model_path)


def model_path_for(model_name, model_dir):
    """Вернуть путь выбранной модели и отклонить неизвестный алгоритм."""

    if model_name not in MODEL_FILENAMES:
        raise ValueError(f"Неизвестная модель: {model_name}")
    return Path(model_dir) / MODEL_FILENAMES[model_name]


def available_models(model_dir):
    """Вернуть метаданные существующих файлов моделей для интерфейса."""

    result = []
    for model_name, filename in MODEL_FILENAMES.items():
        path = Path(model_dir) / filename
        if path.exists():
            result.append(
                {
                    "name": model_name,
                    "display_name": MODEL_DISPLAY_NAMES[model_name],
                    "path": str(path),
                }
            )
    return result


def load_model_registry(registry_path):
    """Прочитать реестр сравнительного обучения или вернуть пустой словарь."""

    path = Path(registry_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_model_name(registry_path, default="random_forest"):
    """Определить автоматически выбранную модель по реестру эксперимента."""

    registry = load_model_registry(registry_path)
    selected = registry.get("selected_model", default)
    return selected if selected in MODEL_FILENAMES else default


def calculate_metrics(y_true, y_pred):
    """Рассчитать MAE, RMSE, MAPE и коэффициент детерминации R²."""

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mape": round(
            float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            2,
        ),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def validate_input(payload):
    """Проверить и типизировать признаки, введенные пользователем."""

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
            except (TypeError, ValueError):
                errors.append(f"Поле {column} должно быть числом")
        else:
            cleaned[column] = str(value).strip()
    if errors:
        raise ValueError("; ".join(errors))
    cleaned["year"] = int(cleaned["year"])
    cleaned["student_count"] = int(cleaned["student_count"])
    return cleaned


def predict_price(model_path, payload, model_name=None):
    """Рассчитать цену выбранной моделью и сформировать рекомендацию."""

    cleaned = validate_input(payload)
    model = load_model(model_path)
    input_frame = pd.DataFrame([cleaned], columns=FEATURE_COLUMNS)
    predicted_price = float(model.predict(input_frame)[0])
    recommendation = build_price_recommendation(predicted_price, cleaned)
    return {
        "predicted_price": round(predicted_price, 2),
        "recommendation": recommendation,
        "input": cleaned,
        "model_name": model_name,
    }


def build_price_recommendation(predicted_price, payload):
    """Сопоставить прогноз с предыдущей ценой и реальным рыночным ориентиром."""

    competitor_price = float(payload.get("competitor_price", predicted_price))
    base_price = float(payload.get("base_price", predicted_price))
    ratio_to_competitor = predicted_price / competitor_price if competitor_price else 1
    ratio_to_base = predicted_price / base_price if base_price else 1

    if ratio_to_competitor > 1.12:
        return "Прогноз заметно выше медианы других вузов по направлению; перед повышением цены следует проверить конкурентное позиционирование."
    if ratio_to_competitor < 0.92:
        return "Прогноз ниже медианы других вузов по направлению; возможно, существует резерв для корректировки цены."
    if ratio_to_base > 1.15:
        return "Прогноз предполагает рост более чем на 15% к предыдущей опубликованной цене; требуется дополнительное обоснование."
    return "Прогнозная цена находится в допустимом диапазоне; рекомендуется использовать ее как базовый ориентир для планирования."


def dataset_summary(dataset_path):
    """Вернуть основные характеристики реального набора мониторинга."""

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
        "total_students": int(df["student_count"].sum()),
        "organizations": int(df["organization"].nunique()),
        "programs": int(df["program_name"].nunique()),
    }
    return summary


def price_dynamics(dataset_path):
    df = read_dataset(dataset_path)
    grouped = df.groupby("year", as_index=False).agg(
        mean_price=("final_price", "mean"),
        mean_students=("student_count", "mean"),
        mean_competitor_price=("competitor_price", "mean"),
        observations=("final_price", "count"),
    )
    grouped["mean_price"] = grouped["mean_price"].round(2)
    grouped["mean_students"] = grouped["mean_students"].round(2)
    grouped["mean_competitor_price"] = grouped["mean_competitor_price"].round(2)
    grouped["growth_percent"] = grouped["mean_price"].pct_change().fillna(0).mul(100).round(2)
    return grouped.to_dict(orient="records")


def program_rating(dataset_path, limit=10):
    df = read_dataset(dataset_path)
    grouped = df.groupby("program_name", as_index=False).agg(
        mean_price=("final_price", "mean"),
        mean_admission_score=("admission_score", "mean"),
        students=("student_count", "sum"),
    )
    grouped["mean_price"] = grouped["mean_price"].round(2)
    grouped["mean_admission_score"] = grouped["mean_admission_score"].round(2)
    grouped = grouped.sort_values(
        ["students", "mean_admission_score"],
        ascending=False,
    ).head(limit)
    return grouped.to_dict(orient="records")


def category_analysis(dataset_path, column):
    df = read_dataset(dataset_path)
    grouped = df.groupby(column, as_index=False).agg(
        mean_price=("final_price", "mean"),
        min_price=("final_price", "min"),
        max_price=("final_price", "max"),
        mean_competitor_price=("competitor_price", "mean"),
        mean_admission_score=("admission_score", "mean"),
        observations=("final_price", "count"),
    )
    for price_column in [
        "mean_price",
        "min_price",
        "max_price",
        "mean_competitor_price",
        "mean_admission_score",
    ]:
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
        mean_admission_score=("admission_score", "mean"),
    )
    grouped["mean_price"] = grouped["mean_price"].round(2)
    grouped["mean_admission_score"] = grouped["mean_admission_score"].round(2)
    grouped = grouped.sort_values("applications", ascending=False).head(12)
    return grouped.to_dict(orient="records")


def dataset_preview(dataset_path, limit=15):
    df = read_dataset(dataset_path).head(limit)
    return df.to_dict(orient="records"), list(df.columns)


def cross_validate_model(dataset_path, model_name="random_forest"):
    """Выполнить пятикратную кросс-валидацию базовой версии модели."""

    df = read_dataset(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    model = build_model(model_name=model_name)
    return _cross_validation_metrics(model, X, y)


def prepare_forecast_payload_from_service(service, year=None):
    """Преобразовать карточку услуги в актуальную схему признаков модели."""

    latest_price = service.last_price()
    year = year or datetime.now().year + 1
    latest_demand = sorted(service.demand_indicators, key=lambda item: item.year, reverse=True)[0] if service.demand_indicators else None
    latest_competitor = sorted(service.competitor_prices, key=lambda item: item.year, reverse=True)[0] if service.competitor_prices else None
    return {
        "year": year,
        "organization": "Текущая образовательная организация",
        "program_name": service.program.name,
        "base_price": latest_price,
        "competitor_price": latest_competitor.price if latest_competitor else latest_price * 0.98,
        "student_count": 60,
        "admission_score": latest_demand.demand_index if latest_demand else 65,
        "salary_index": latest_demand.salary_index if latest_demand else 160,
    }
