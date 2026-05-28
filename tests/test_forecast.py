from app.ml_service import predict_price
from config import Config
from tests.test_auth import login


FORECAST_PAYLOAD = {
    "year": 2027,
    "region": "Москва",
    "education_level": "бакалавриат",
    "program_name": "Искусственный интеллект и анализ данных",
    "study_format": "дистанционная",
    "duration_months": 48,
    "base_price": 150000,
    "competitor_price": 145000,
    "demand_index": 72,
    "salary_index": 76,
    "advertising_spend": 160000,
    "discount_percent": 8,
    "student_count": 75,
}


def test_prediction_positive():
    result = predict_price(Config.MODEL_PATH, FORECAST_PAYLOAD)
    assert result["predicted_price"] > 0
    assert "recommendation" in result


def test_forecast_page_saves_result_and_cabinet_shows_it(app_instance):
    client = app_instance.test_client()
    login(client, "analyst", "analyst123")

    response = client.post("/forecast", data=FORECAST_PAYLOAD, follow_redirects=True)
    assert response.status_code == 200
    assert "Прогнозная цена".encode("utf-8") in response.data

    response = client.get("/dashboard/forecasts")
    assert response.status_code == 200
    assert "Искусственный интеллект".encode("utf-8") in response.data
