"""Запуск сравнительного обучения моделей цен.

Автор: Горячевская Екатерина Николевна
Тема ВКР: «Анализ и прогнозирование ценообразования на образовательные услуги».
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.ml_service import train_all_models
from config import Config

if __name__ == "__main__":
    registry = train_all_models(
        Config.PROCESSED_DATASET_PATH,
        Config.MODEL_DIR,
        registry_path=Config.MODEL_REGISTRY_PATH,
        diagnostics_dir=Config.DIAGNOSTICS_DIR,
    )
    print("Сравнительное обучение завершено.")
    print("Выбранная модель:", registry["selected_model"])
    for name, report in registry["models"].items():
        print(name, report["metrics"]["test"])
