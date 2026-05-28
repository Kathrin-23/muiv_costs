from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from config import Config


db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Для доступа к разделу необходимо войти в систему."
login_manager.login_message_category = "warning"


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)

    from app.models import Role, User, EducationalProgram, EducationalService, PriceHistory, DemandIndicator, CompetitorPrice, DatasetFile, ForecastResult, SystemSetting, FeedbackMessage
    from app.routes import register_routes

    register_routes(app)

    with app.app_context():
        db.create_all()
        seed_reference_data()

    return app


@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    return User.query.get(int(user_id))


def make_password_hash(password):
    return generate_password_hash(password, method="pbkdf2:sha256")


def seed_reference_data():
    from app.models import Role, User, EducationalProgram, EducationalService, PriceHistory, DemandIndicator, CompetitorPrice, DatasetFile, SystemSetting
    from app import db
    from config import Config
    import os
    import pandas as pd

    role_titles = {
        "admin": "Администратор",
        "analyst": "Аналитик",
        "user": "Пользователь",
    }
    for name, title in role_titles.items():
        role = Role.query.filter_by(name=name).first()
        if role is None:
            db.session.add(Role(name=name, title=title))
        elif role.title != title:
            role.title = title
    db.session.commit()

    roles = {role.name: role for role in Role.query.all()}
    demo_users = [
        ("admin", "Администратор системы", "admin@example.local", "admin", "admin123"),
        ("analyst", "Аналитик по ценообразованию", "analyst@example.local", "analyst", "analyst123"),
        ("user", "Пользователь системы", "user@example.local", "user", "user123"),
    ]
    for username, full_name, email, role_name, password in demo_users:
        user = User.query.filter_by(username=username).first()
        if user is None:
            db.session.add(User(
                username=username,
                full_name=full_name,
                email=email,
                role_id=roles[role_name].id,
                password_hash=make_password_hash(password),
                is_active=True,
            ))
        else:
            user.full_name = full_name
            user.email = email
            user.role_id = roles[role_name].id
            user.is_active = True
            if user.password_hash.startswith("scrypt:"):
                user.password_hash = make_password_hash(password)
    db.session.commit()

    if SystemSetting.query.count() == 0:
        db.session.add_all([
            SystemSetting(key="forecast_horizon", value="1", description="Горизонт прогноза в годах"),
            SystemSetting(key="default_currency", value="RUB", description="Валюта расчетов"),
            SystemSetting(key="model_type", value="RandomForestRegressor", description="Используемый алгоритм прогнозирования"),
            SystemSetting(key="min_dataset_rows", value="50", description="Минимальный объем набора данных"),
            SystemSetting(key="report_org", value="Кафедра информационных систем", description="Подразделение для отчета"),
        ])
        db.session.commit()

    if EducationalProgram.query.count() == 0:
        programs = [
            EducationalProgram(name="Прикладная информатика", education_level="бакалавриат", department="Кафедра информационных систем", duration_months=48),
            EducationalProgram(name="Искусственный интеллект и анализ данных", education_level="бакалавриат", department="Кафедра информационных систем", duration_months=48),
            EducationalProgram(name="Бизнес-аналитика", education_level="магистратура", department="Кафедра информационных систем", duration_months=24),
            EducationalProgram(name="Разработка веб-приложений", education_level="дополнительное образование", department="Центр дополнительного образования", duration_months=6),
        ]
        db.session.add_all(programs)
        db.session.commit()

    if EducationalService.query.count() == 0:
        program_map = {program.name: program for program in EducationalProgram.query.all()}
        services = [
            EducationalService(program_id=program_map["Прикладная информатика"].id, name="Обучение по программе бакалавриата", study_format="очная", region="Москва", current_price=196000),
            EducationalService(program_id=program_map["Искусственный интеллект и анализ данных"].id, name="Обучение по программе бакалавриата", study_format="дистанционная", region="Москва", current_price=154000),
            EducationalService(program_id=program_map["Бизнес-аналитика"].id, name="Обучение по программе магистратуры", study_format="очно-заочная", region="Санкт-Петербург", current_price=208000),
            EducationalService(program_id=program_map["Разработка веб-приложений"].id, name="Курс дополнительного образования", study_format="дистанционная", region="Казань", current_price=54000),
        ]
        db.session.add_all(services)
        db.session.commit()

    if PriceHistory.query.count() == 0:
        for service in EducationalService.query.all():
            for year in range(2021, 2027):
                multiplier = 0.82 + (year - 2021) * 0.055
                db.session.add(PriceHistory(service_id=service.id, year=year, price=round(service.current_price * multiplier, 2), discount_percent=5 + (year % 3), student_count=40 + year % 7 * 6))
                db.session.add(DemandIndicator(service_id=service.id, year=year, demand_index=55 + year % 5 * 7, salary_index=60 + year % 4 * 4, advertising_spend=100000 + year % 6 * 25000))
                db.session.add(CompetitorPrice(service_id=service.id, competitor_name="Среднее значение по рынку", year=year, price=round(service.current_price * multiplier * 0.97, 2), source="Демонстрационный набор данных"))
        db.session.commit()

    if DatasetFile.query.count() == 0 and os.path.exists(Config.RAW_DATASET_PATH):
        try:
            df = pd.read_csv(Config.RAW_DATASET_PATH)
            db.session.add(DatasetFile(filename="educational_services_prices.csv", original_name="educational_services_prices.csv", row_count=len(df), status="загружен", description="Демонстрационный набор данных по образовательным услугам"))
            db.session.commit()
        except Exception:
            db.session.rollback()
