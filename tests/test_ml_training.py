"""Тесты сравнительного обучения и артефактов Colab."""

import json
from pathlib import Path

import pandas as pd

from app.ml_service import (
    FEATURE_COLUMNS,
    MODEL_FILENAMES,
    MODEL_PARAMETER_GRIDS,
    calculate_metrics,
    load_model,
    read_dataset,
    split_dataset,
    train_all_models,
)
from config import Config
from tests.test_auth import login


def test_comparison_training_saves_three_models_and_metrics(monkeypatch, tmp_path):
    """Облегченная сетка сохраняет тот же полный набор рабочих артефактов."""

    source = pd.read_csv(Config.PROCESSED_DATASET_PATH).sample(
        n=150,
        random_state=42,
    )
    dataset_path = tmp_path / "dataset.csv"
    source.to_csv(dataset_path, index=False)
    model_dir = tmp_path / "models"

    small_grids = {
        "random_forest": {
            "regressor__n_estimators": [20],
            "regressor__max_depth": [6],
            "regressor__min_samples_leaf": [1],
        },
        "gradient_boosting": {
            "regressor__n_estimators": [20],
            "regressor__learning_rate": [0.1],
            "regressor__max_depth": [2],
        },
        "ridge": {"regressor__alpha": [1.0]},
    }
    for model_name, grid in small_grids.items():
        monkeypatch.setitem(MODEL_PARAMETER_GRIDS, model_name, grid)

    registry = train_all_models(
        dataset_path,
        model_dir,
        registry_path=model_dir / "model_registry.json",
        diagnostics_dir=model_dir / "diagnostics",
    )

    assert set(registry["models"]) == set(MODEL_FILENAMES)
    assert registry["selected_model"] in MODEL_FILENAMES
    assert (model_dir / "model_comparison.csv").exists()
    for model_name, filename in MODEL_FILENAMES.items():
        assert (model_dir / filename).exists()
        assert (model_dir / f"metrics_{model_name}.json").exists()
        load_model(model_dir / filename)
        report = registry["models"][model_name]
        assert set(report["metrics"]) == {"train", "validation", "test"}
        assert report["cross_validation"]["folds"] == 5
        assert report["split_years"] == {
            "train": [2019, 2020, 2021, 2022],
            "validation": [2023],
            "test": [2024],
        }
        assert report["best_params"]
        assert "conclusion" in report["overfitting"]


def test_training_page_displays_comparison_registry(app_instance):
    client = app_instance.test_client()
    login(client, "analyst", "analyst123")

    response = client.get("/model-training")

    assert response.status_code == 200
    assert "Сравнение моделей".encode("utf-8") in response.data
    assert "Случайный лес".encode("utf-8") in response.data
    assert "Градиентный бустинг".encode("utf-8") in response.data
    assert "Ridge".encode("utf-8") in response.data
    assert "валидационная".encode("utf-8") in response.data


def test_colab_notebook_has_executed_outputs():
    notebook_path = Path(__file__).resolve().parents[1] / "ml" / "train_model.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["metadata"]["colab"]["include_colab_link"] is True
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "colab.research.google.com" in source
    assert "DATASET_URL" in source
    assert "GridSearchCV" in source
    assert "TimeSeriesSplit" in source
    assert 'clean["year"] <= 2022' in source
    assert "advertising_spend" not in source
    assert "discount_percent" not in source
    assert "model_random_forest.pkl" in source
    assert "model_gradient_boosting.pkl" in source
    assert "model_ridge.pkl" in source

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_dataset_size_variability_and_temporal_split():
    dataset = read_dataset(Config.PROCESSED_DATASET_PATH)

    assert dataset.shape[0] >= 1_000
    assert dataset["organization"].nunique() > 100
    assert dataset["program_name"].nunique() > 20
    assert dataset["student_count"].nunique() > 100
    assert dataset["admission_score"].nunique() > 100
    assert FEATURE_COLUMNS == [
        "year",
        "organization",
        "program_name",
        "base_price",
        "competitor_price",
        "student_count",
        "admission_score",
        "salary_index",
    ]

    splits = split_dataset(dataset)
    assert sorted(splits["train"][0]["year"].unique()) == [2019, 2020, 2021, 2022]
    assert sorted(splits["validation"][0]["year"].unique()) == [2023]
    assert sorted(splits["test"][0]["year"].unique()) == [2024]


def test_competitor_price_excludes_current_organization():
    raw_path = Path(Config.PROCESSED_DATASET_PATH).parents[1] / "raw" / "educational_services_prices.csv"
    raw = pd.read_csv(raw_path)
    for _, row in raw.sample(100, random_state=42).iterrows():
        peers = raw[
            (raw["year"] == row["year"])
            & (raw["program_name"] == row["program_name"])
            & (raw["organization"] != row["organization"])
        ]
        assert not peers.empty
        assert row["competitor_price"] == peers["final_price"].median()


def test_mape_is_stored_as_percent():
    result = calculate_metrics([100, 200], [90, 220])
    assert result["mape"] == 10.0
