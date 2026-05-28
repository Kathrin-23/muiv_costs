import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app, db

app = create_app()

with app.app_context():
    db.drop_all()
    db.create_all()
    from app import seed_reference_data
    seed_reference_data()
    print("База данных создана и заполнена демонстрационными данными.")
