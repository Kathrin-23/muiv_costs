"""Пример прогнозирования цены сохраненной моделью.

Автор: Горячевская Екатерина Николевна
Тема ВКР: «Анализ и прогнозирование ценообразования на образовательные услуги».
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.ml_service import model_path_for, predict_price, selected_model_name
from config import Config

payload = {
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

if __name__ == "__main__":
    model_name = selected_model_name(Config.MODEL_REGISTRY_PATH)
    model_path = model_path_for(model_name, Config.MODEL_DIR)
    result = predict_price(model_path, payload, model_name=model_name)
    print(result)
