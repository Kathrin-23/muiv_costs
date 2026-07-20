from io import BytesIO

from app.ml_service import dataset_summary, price_dynamics
from config import Config
from tests.test_auth import login
from tests.test_forecast import FORECAST_PAYLOAD


def test_dataset_summary_and_price_dynamics():
    summary = dataset_summary(Config.PROCESSED_DATASET_PATH)
    assert summary["rows"] > 0
    assert summary["max_year"] >= summary["min_year"]
    assert summary["min_price"] <= summary["max_price"]
    dynamics = price_dynamics(Config.PROCESSED_DATASET_PATH)
    assert len(dynamics) > 0
    assert "growth_percent" in dynamics[0]


def test_dataset_upload_and_preview(app_instance, monkeypatch, tmp_path):
    upload_dir = tmp_path / "upload_case"
    processed = tmp_path / "processed.csv"
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(upload_dir))
    monkeypatch.setattr(Config, "PROCESSED_DATASET_PATH", str(processed))

    client = app_instance.test_client()
    login(client, "analyst", "analyst123")

    csv_data = (
        "year,organization,program_name,base_price,competitor_price,student_count,"
        "admission_score,salary_index,final_price\n"
        "2026,Тестовый вуз,Экономика,100000,98000,80,70,165,105000\n"
    )
    response = client.post(
        "/upload-dataset",
        data={"dataset": (BytesIO(csv_data.encode("utf-8")), "test_dataset.csv"), "description": "pytest"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "test_dataset.csv".encode("utf-8") in response.data
    assert processed.exists()


def test_dataset_upload_by_url_route(app_instance, monkeypatch, tmp_path):
    """Маршрут принимает URL и оставляет локальную обработку общей."""

    source = tmp_path / "from_url.csv"
    processed = tmp_path / "processed_from_url.csv"
    source.write_text(
        "year,organization,program_name,base_price,competitor_price,student_count,"
        "admission_score,salary_index,final_price\n"
        "2026,URL вуз,Экономика,100000,98000,80,70,165,105000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Config, "PROCESSED_DATASET_PATH", str(processed))
    monkeypatch.setattr(
        "app.routes.download_dataset_from_url",
        lambda url, folder, max_bytes: (source, 1),
    )
    client = app_instance.test_client()
    login(client, "analyst", "analyst123")

    response = client.post(
        "/upload-dataset",
        data={"dataset_url": "https://example.org/prices.csv", "description": "URL"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "https://example.org/prices.csv".encode("utf-8") in response.data
    assert processed.exists()


def test_export_report_and_admin_pages(app_instance):
    client = app_instance.test_client()
    login(client, "admin", "admin123")
    client.post("/forecast", data=FORECAST_PAYLOAD, follow_redirects=True)

    response = client.get("/reports/export/xlsx")
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].startswith("attachment")

    for path in ["/admin/users", "/admin/roles", "/admin/services", "/admin/datasets", "/admin/reports", "/dashboard/reports"]:
        response = client.get(path)
        assert response.status_code == 200
