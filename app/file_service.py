"""Загрузка и подготовка файлов с наборами данных.

Автор: Горячевская Екатерина Николевна
Тема ВКР: «Анализ и прогнозирование ценообразования на образовательные услуги».

Модуль поддерживает два равноправных сценария: загрузку CSV/XLSX из браузера
и получение такого же файла по общедоступной HTTP(S)-ссылке.
"""

from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from werkzeug.utils import secure_filename

from app.ml_service import normalize_dataset

ALLOWED_EXTENSIONS = {"csv", "xlsx"}
CONTENT_TYPE_EXTENSIONS = {
    "text/csv": ".csv",
    "application/csv": ".csv",
    "application/vnd.ms-excel": ".xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}
DEFAULT_DOWNLOAD_LIMIT = 16 * 1024 * 1024


def allowed_file(filename):
    """Проверить, имеет ли файл поддерживаемое расширение."""

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_tabular_file(path):
    """Прочитать CSV или XLSX в ``DataFrame`` по расширению файла."""

    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Поддерживаются только файлы CSV и XLSX")


def save_uploaded_file(file_storage, upload_folder):
    """Сохранить локально загруженный файл и вернуть путь и число строк."""

    if not file_storage or not file_storage.filename:
        raise ValueError("Файл не выбран")
    if not allowed_file(file_storage.filename):
        raise ValueError("Поддерживаются только файлы CSV и XLSX")
    filename = secure_filename(file_storage.filename)
    target_dir = Path(upload_folder)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    file_storage.save(target_path)
    return target_path, count_rows(target_path)


def _extension_from_response(url, response):
    """Определить расширение по URL, Content-Disposition или Content-Type."""

    url_name = Path(unquote(urlparse(url).path)).name
    if allowed_file(url_name):
        return Path(url_name).suffix.lower()

    disposition = response.headers.get("Content-Disposition", "")
    if "filename=" in disposition:
        response_name = disposition.split("filename=", 1)[1].strip(" \"'")
        if allowed_file(response_name):
            return Path(response_name).suffix.lower()

    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    extension = CONTENT_TYPE_EXTENSIONS.get(content_type)
    if extension:
        return extension
    raise ValueError("Ссылка должна вести на файл CSV или XLSX")


def download_dataset_from_url(url, upload_folder, max_bytes=DEFAULT_DOWNLOAD_LIMIT):
    """Скачать CSV/XLSX по HTTP(S)-ссылке и сохранить в каталоге загрузок.

    Ответ читается частями, поэтому ограничение размера проверяется и по
    заголовку ``Content-Length``, и во время фактического скачивания.
    """

    normalized_url = (url or "").strip()
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Укажите корректную ссылку HTTP(S) на CSV или XLSX")

    response = requests.get(
        normalized_url,
        stream=True,
        timeout=(10, 60),
        allow_redirects=True,
        headers={"User-Agent": "muiv-costs-dataset-loader/1.0"},
    )
    response.raise_for_status()

    declared_size = int(response.headers.get("Content-Length", 0) or 0)
    if declared_size > max_bytes:
        raise ValueError("Размер файла по ссылке превышает 16 МБ")

    extension = _extension_from_response(normalized_url, response)
    target_dir = Path(upload_folder)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = target_dir / f"dataset_from_url_{timestamp}{extension}"

    downloaded = 0
    try:
        with target_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError("Размер файла по ссылке превышает 16 МБ")
                output.write(chunk)
        row_count = count_rows(target_path)
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    return target_path, row_count


def count_rows(path):
    """Вернуть число строк данных без учета заголовка."""

    return len(read_tabular_file(path))


def preview_file(path, limit=10):
    """Вернуть первые строки и названия столбцов для предпросмотра."""

    df = read_tabular_file(path).head(limit)
    return df.to_dict(orient="records"), df.columns.tolist()


def copy_dataset_to_processed(source_path, destination_path):
    """Нормализовать исходный файл и сохранить рабочую CSV-копию."""

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_dataset(read_tabular_file(source_path))
    normalized.to_csv(destination_path, index=False, encoding="utf-8-sig")
    return destination_path
