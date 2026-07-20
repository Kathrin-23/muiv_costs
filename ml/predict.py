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
    "year": 2025,
    "organization": "НИУ ВШЭ, г. Москва",
    "program_name": "Экономика",
    "base_price": 350000,
    "competitor_price": 230000,
    "student_count": 100,
    "admission_score": 75,
    "salary_index": 163.6,
}

if __name__ == "__main__":
    model_name = selected_model_name(Config.MODEL_REGISTRY_PATH)
    model_path = model_path_for(model_name, Config.MODEL_DIR)
    result = predict_price(model_path, payload, model_name=model_name)
    print(result)
