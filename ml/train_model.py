from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.ml_service import train_model, cross_validate_model
from config import Config

if __name__ == "__main__":
    metrics_path = Path(Config.MODEL_PATH).with_name("metrics.json")
    metrics = train_model(Config.PROCESSED_DATASET_PATH, Config.MODEL_PATH, metrics_path, model_name="random_forest")
    cv = cross_validate_model(Config.PROCESSED_DATASET_PATH, model_name="random_forest")
    print("Метрики обучения:", metrics)
    print("Кросс-валидация:", cv)
