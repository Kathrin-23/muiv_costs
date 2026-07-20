from app.ml_service import predict_price
from app.models import ForecastResult
from config import Config
from tests.test_auth import login


FORECAST_PAYLOAD = {
    "year": 2025,
    "organization": "НИУ ВШЭ, г. Москва",
    "program_name": "Экономика",
    "base_price": 350000,
    "competitor_price": 230000,
    "student_count": 100,
    "admission_score": 75,
    "salary_index": 163.6,
}


def test_prediction_positive():
    result = predict_price(Config.MODEL_PATH, FORECAST_PAYLOAD)
    assert result["predicted_price"] > 0
    assert "recommendation" in result


def test_forecast_form_contains_only_supported_features(app_instance):
    client = app_instance.test_client()
    login(client, "analyst", "analyst123")

    response = client.get("/forecast")

    assert response.status_code == 200
    for field in FORECAST_PAYLOAD:
        assert f'name="{field}"'.encode() in response.data
    for removed_field in [
        "region",
        "advertising_spend",
        "discount_percent",
        "demand_index",
        "study_format",
        "duration_months",
    ]:
        assert f'name="{removed_field}"'.encode() not in response.data


def test_forecast_page_saves_result_and_cabinet_shows_it(app_instance):
    client = app_instance.test_client()
    login(client, "analyst", "analyst123")

    payload = {**FORECAST_PAYLOAD, "model_name": "gradient_boosting"}
    response = client.post("/forecast", data=payload, follow_redirects=True)
    assert response.status_code == 200
    assert "Прогнозная цена".encode("utf-8") in response.data
    assert "Градиентный бустинг".encode("utf-8") in response.data

    with app_instance.app_context():
        saved = ForecastResult.query.order_by(ForecastResult.id.desc()).first()
        assert saved.model_name == "gradient_boosting"

    response = client.get("/dashboard/forecasts")
    assert response.status_code == 200
    assert "Экономика".encode("utf-8") in response.data
    assert "НИУ ВШЭ".encode("utf-8") in response.data
    assert b"gradient_boosting" in response.data
