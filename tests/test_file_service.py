"""Тесты загрузки наборов данных из файла и по URL."""

from io import BytesIO

import pandas as pd
import pytest

from app.file_service import download_dataset_from_url, save_uploaded_file


class FakeResponse:
    """Минимальная имитация потокового HTTP-ответа для теста."""

    def __init__(self, content, content_type="text/csv"):
        self.content = content
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(content)),
        }

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.content


def test_download_csv_by_url(monkeypatch, tmp_path):
    content = b"year,final_price\n2024,120000\n2025,135000\n"
    monkeypatch.setattr(
        "app.file_service.requests.get",
        lambda *args, **kwargs: FakeResponse(content),
    )

    path, row_count = download_dataset_from_url(
        "https://example.org/prices.csv",
        tmp_path,
    )

    assert path.exists()
    assert path.suffix == ".csv"
    assert row_count == 2
    assert pd.read_csv(path)["final_price"].tolist() == [120000, 135000]


def test_download_rejects_non_http_url(tmp_path):
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        download_dataset_from_url("file:///etc/passwd", tmp_path)


def test_local_upload_is_still_supported(tmp_path):
    from werkzeug.datastructures import FileStorage

    storage = FileStorage(
        stream=BytesIO(b"year,final_price\n2025,140000\n"),
        filename="prices.csv",
    )
    path, row_count = save_uploaded_file(storage, tmp_path)

    assert path.name == "prices.csv"
    assert row_count == 1
