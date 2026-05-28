from pathlib import Path
from werkzeug.utils import secure_filename
import pandas as pd

from app.ml_service import normalize_dataset

ALLOWED_EXTENSIONS = {"csv", "xlsx"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file_storage, upload_folder):
    if not file_storage or not file_storage.filename:
        raise ValueError("Файл не выбран")
    if not allowed_file(file_storage.filename):
        raise ValueError("Поддерживаются только файлы CSV и XLSX")
    filename = secure_filename(file_storage.filename)
    target_dir = Path(upload_folder)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    file_storage.save(target_path)
    row_count = count_rows(target_path)
    return target_path, row_count


def count_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return len(pd.read_excel(path))
    return len(pd.read_csv(path))


def preview_file(path, limit=10):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path).head(limit)
    else:
        df = pd.read_csv(path).head(limit)
    return df.to_dict(orient="records"), df.columns.tolist()


def copy_dataset_to_processed(source_path, destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(source_path)
    else:
        df = pd.read_csv(source_path)
    normalized = normalize_dataset(df)
    normalized.to_csv(destination_path, index=False, encoding="utf-8-sig")
    return destination_path
