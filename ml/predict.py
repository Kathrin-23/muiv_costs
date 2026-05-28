from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.ml_service import predict_price
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
    result = predict_price(Config.MODEL_PATH, payload)
    print(result)
