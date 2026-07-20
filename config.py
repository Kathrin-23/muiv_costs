from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"
DATA_DIR = BASE_DIR / "data"
MODEL_ARTIFACT_DIR = BASE_DIR / "ml"

DATABASE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

class Config:
    SECRET_KEY = "education-pricing-demo-secret-key"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_DIR / 'education_pricing.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    EXPORT_FOLDER = str(EXPORT_DIR)
    # Совместимый путь по умолчанию; интерфейс выбирает модель через реестр.
    MODEL_PATH = str(MODEL_ARTIFACT_DIR / "model_random_forest.pkl")
    MODEL_DIR = str(MODEL_ARTIFACT_DIR)
    MODEL_REGISTRY_PATH = str(MODEL_ARTIFACT_DIR / "model_registry.json")
    MODEL_COMPARISON_PATH = str(MODEL_ARTIFACT_DIR / "model_comparison.csv")
    DIAGNOSTICS_DIR = str(MODEL_ARTIFACT_DIR / "diagnostics")
    RAW_DATASET_PATH = str(DATA_DIR / "raw" / "educational_services_prices.csv")
    PROCESSED_DATASET_PATH = str(DATA_DIR / "processed" / "prepared_dataset.csv")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
