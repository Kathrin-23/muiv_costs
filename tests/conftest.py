import pytest

from app import create_app
from config import Config


@pytest.fixture
def app_instance(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(Config, "EXPORT_FOLDER", str(tmp_path / "exports"))
    app = create_app()
    app.config.update(TESTING=True)
    return app
