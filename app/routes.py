"""Маршруты и сценарии взаимодействия пользователя с приложением.

Маршруты получают данные форм, передают файловую обработку в ``file_service``,
обучение и прогнозирование — в ``ml_service``, а затем сохраняют результаты
через модели SQLAlchemy. Основные ML-сценарии снабжены отдельными docstring.
"""

import json
from datetime import datetime
from pathlib import Path

from flask import flash, redirect, render_template, request, send_file, send_from_directory, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash

from app import db, make_password_hash
from app.file_service import (
    copy_dataset_to_processed,
    download_dataset_from_url,
    save_uploaded_file,
)
from app.ml_service import (
    MODEL_DISPLAY_NAMES,
    available_models,
    category_analysis,
    competitor_analysis,
    dataset_preview,
    dataset_summary,
    demand_analysis,
    load_model_registry,
    model_path_for,
    prepare_forecast_payload_from_service,
    predict_price,
    price_dynamics,
    program_rating,
    selected_model_name,
    train_all_models,
)
from app.models import (
    CompetitorPrice,
    DatasetFile,
    DemandIndicator,
    EducationalProgram,
    EducationalService,
    FeedbackMessage,
    ForecastResult,
    PriceHistory,
    ReportFile,
    Role,
    SystemSetting,
    User,
)
from app.report_service import (
    export_dataset_summary_to_docx,
    export_dataset_summary_to_xlsx,
    export_forecasts_to_docx,
    export_forecasts_to_xlsx,
)
from config import Config


def register_routes(app):
    app.context_processor(inject_layout_context)
    app.add_url_rule("/", "index", index)
    app.add_url_rule("/login", "login", login, methods=["GET", "POST"])
    app.add_url_rule("/register", "register", register, methods=["GET", "POST"])
    app.add_url_rule("/logout", "logout", logout)
    app.add_url_rule("/dashboard", "dashboard", dashboard)
    app.add_url_rule("/dashboard/forecasts", "user_forecasts", user_forecasts)
    app.add_url_rule("/dashboard/reports", "user_reports", user_reports)
    app.add_url_rule("/dashboard/datasets", "user_datasets", user_datasets)
    app.add_url_rule("/dashboard/profile", "profile_settings", profile_settings, methods=["GET", "POST"])
    app.add_url_rule("/services", "services", services)
    app.add_url_rule("/services/<int:service_id>", "service_card", service_card)
    app.add_url_rule("/prices", "price_management", price_management, methods=["GET", "POST"])
    app.add_url_rule("/upload-dataset", "upload_dataset", upload_dataset, methods=["GET", "POST"])
    app.add_url_rule("/datasets/<int:dataset_id>/delete", "delete_dataset", delete_dataset)
    app.add_url_rule("/price-analysis", "price_analysis", price_analysis)
    app.add_url_rule("/forecast", "forecast", forecast, methods=["GET", "POST"])
    app.add_url_rule("/forecast/service/<int:service_id>", "forecast_service", forecast_service)
    app.add_url_rule("/model-training", "model_training", model_training, methods=["GET", "POST"])
    app.add_url_rule("/model-diagnostics/<path:filename>", "model_diagnostic", model_diagnostic)
    app.add_url_rule("/reports", "reports", reports)
    app.add_url_rule("/reports/export/xlsx", "export_xlsx", export_xlsx)
    app.add_url_rule("/reports/export/docx", "export_docx", export_docx)
    app.add_url_rule("/reports/export/dataset", "export_dataset", export_dataset)
    app.add_url_rule("/reports/export/analysis-docx", "export_analysis_docx", export_analysis_docx)
    app.add_url_rule("/help", "help_page", help_page)
    app.add_url_rule("/feedback", "feedback", feedback, methods=["GET", "POST"])
    app.add_url_rule("/admin", "admin_panel", admin_panel)
    app.add_url_rule("/admin/users", "admin_users", admin_users, methods=["GET", "POST"])
    app.add_url_rule("/admin/users/<int:user_id>/toggle", "admin_toggle_user", admin_toggle_user)
    app.add_url_rule("/admin/roles", "admin_roles", admin_roles, methods=["GET", "POST"])
    app.add_url_rule("/admin/programs", "admin_programs", admin_programs, methods=["GET", "POST"])
    app.add_url_rule("/admin/programs/<int:program_id>/edit", "admin_edit_program", admin_edit_program, methods=["GET", "POST"])
    app.add_url_rule("/admin/programs/<int:program_id>/delete", "admin_delete_program", admin_delete_program)
    app.add_url_rule("/admin/services", "admin_services", admin_services, methods=["GET", "POST"])
    app.add_url_rule("/admin/services/<int:service_id>/edit", "admin_edit_service", admin_edit_service, methods=["GET", "POST"])
    app.add_url_rule("/admin/services/<int:service_id>/delete", "admin_delete_service", admin_delete_service)
    app.add_url_rule("/admin/datasets", "admin_datasets", admin_datasets)
    app.add_url_rule("/admin/reports", "admin_reports", admin_reports)
    app.add_url_rule("/admin/reports/<int:report_id>/delete", "admin_delete_report", admin_delete_report)
    app.add_url_rule("/admin/feedback", "admin_feedback", admin_feedback)
    app.add_url_rule("/admin/settings", "admin_settings", admin_settings, methods=["GET", "POST"])
    app.add_url_rule("/api/service/<int:service_id>/history", "api_service_history", api_service_history)


def inject_layout_context():
    names = {
        "index": "Главная",
        "login": "Вход",
        "register": "Регистрация",
        "dashboard": "Личный кабинет",
        "user_forecasts": "Мои прогнозы",
        "user_reports": "Мои отчеты",
        "user_datasets": "Загруженные данные",
        "profile_settings": "Настройки профиля",
        "services": "Услуги",
        "service_card": "Карточка услуги",
        "price_management": "Цены",
        "upload_dataset": "Датасеты",
        "price_analysis": "Анализ цен",
        "forecast": "Прогноз",
        "model_training": "Обучение модели",
        "reports": "Отчеты",
        "help_page": "Справка",
        "feedback": "Обратная связь",
        "admin_panel": "Администрирование",
        "admin_users": "Пользователи",
        "admin_roles": "Роли",
        "admin_programs": "Программы",
        "admin_edit_program": "Редактирование программы",
        "admin_services": "Услуги",
        "admin_edit_service": "Редактирование услуги",
        "admin_datasets": "Датасеты",
        "admin_reports": "Отчеты",
        "admin_feedback": "Обращения",
        "admin_settings": "Настройки",
    }
    endpoint = request.endpoint or "index"
    breadcrumbs = [{"title": "Главная", "url": url_for("index")}]
    if endpoint.startswith("admin_") and endpoint != "admin_panel":
        breadcrumbs.append({"title": "Администрирование", "url": url_for("admin_panel")})
    elif endpoint.startswith("user_") or endpoint == "profile_settings":
        breadcrumbs.append({"title": "Личный кабинет", "url": url_for("dashboard")})
    if endpoint != "index":
        breadcrumbs.append({"title": names.get(endpoint, "Раздел"), "url": None})
    return {
        "breadcrumbs": breadcrumbs,
        "author_name": "Горячевская Екатерина Николевна",
        "project_year": "2026",
    }


def role_required(*role_names):
    def decorator(view_func):
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("login"))
            if not current_user.role or current_user.role.name not in role_names:
                flash("Недостаточно прав для доступа к выбранному разделу.", "danger")
                return redirect(url_for("dashboard"))
            return view_func(*args, **kwargs)
        wrapper.__name__ = view_func.__name__
        return wrapper
    return decorator


def index():
    service_count = EducationalService.query.count()
    forecast_count = ForecastResult.query.count()
    dataset_count = DatasetFile.query.count()
    latest_forecasts = ForecastResult.query.order_by(ForecastResult.created_at.desc()).limit(5).all()
    return render_template(
        "index.html",
        service_count=service_count,
        forecast_count=forecast_count,
        dataset_count=dataset_count,
        latest_forecasts=latest_forecasts,
    )


def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.is_active and check_password_hash(user.password_hash, password):
            user.last_login_at = datetime.utcnow()
            db.session.commit()
            login_user(user)
            flash("Вход выполнен успешно.", "success")
            return redirect(url_for("dashboard"))
        flash("Неверный логин или пароль.", "danger")
    return render_template("login.html")


def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not full_name or not email or not password:
            flash("Заполните все поля регистрации.", "warning")
            return render_template("register.html")
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Пользователь с таким логином или почтой уже существует.", "danger")
            return render_template("register.html")
        role = Role.query.filter_by(name="user").first()
        user = User(username=username, full_name=full_name, email=email, role_id=role.id, password_hash=make_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash("Регистрация завершена. Теперь можно войти в систему.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


def logout():
    logout_user()
    flash("Вы вышли из системы.", "info")
    return redirect(url_for("index"))


@login_required
def dashboard():
    metrics = {
        "programs": EducationalProgram.query.count(),
        "services": EducationalService.query.count(),
        "forecasts": ForecastResult.query.count(),
        "datasets": DatasetFile.query.count(),
    }
    latest_forecasts = ForecastResult.query.filter_by(created_by=current_user.id).order_by(ForecastResult.created_at.desc()).limit(8).all()
    latest_datasets = DatasetFile.query.order_by(DatasetFile.uploaded_at.desc()).limit(5).all()
    return render_template("dashboard.html", metrics=metrics, latest_forecasts=latest_forecasts, latest_datasets=latest_datasets)


@login_required
def user_forecasts():
    forecasts = ForecastResult.query.filter_by(created_by=current_user.id).order_by(ForecastResult.created_at.desc()).all()
    return render_template("user_forecasts.html", forecasts=forecasts)


@login_required
def user_reports():
    reports_list = ReportFile.query.filter_by(created_by=current_user.id).order_by(ReportFile.created_at.desc()).all()
    return render_template("user_reports.html", reports=reports_list)


@login_required
def user_datasets():
    datasets = DatasetFile.query.order_by(DatasetFile.uploaded_at.desc()).all()
    preview_rows, preview_columns = [], []
    try:
        preview_rows, preview_columns = dataset_preview(Config.PROCESSED_DATASET_PATH, limit=10)
    except Exception as exc:
        flash(f"Предпросмотр данных недоступен: {exc}", "warning")
    return render_template("user_datasets.html", datasets=datasets, preview_rows=preview_rows, preview_columns=preview_columns)


@login_required
def profile_settings():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not full_name or not email:
            flash("ФИО и email обязательны.", "warning")
        elif User.query.filter(User.email == email, User.id != current_user.id).first():
            flash("Этот email уже используется другим пользователем.", "danger")
        else:
            current_user.full_name = full_name
            current_user.email = email
            if password:
                current_user.password_hash = make_password_hash(password)
            db.session.commit()
            flash("Профиль обновлен.", "success")
            return redirect(url_for("profile_settings"))
    return render_template("profile_settings.html")


@login_required
def services():
    region = request.args.get("region")
    study_format = request.args.get("study_format")
    education_level = request.args.get("education_level")
    department = request.args.get("department")
    max_price = request.args.get("max_price")
    query = EducationalService.query.join(EducationalProgram)
    if region:
        query = query.filter(EducationalService.region == region)
    if study_format:
        query = query.filter(EducationalService.study_format == study_format)
    if education_level:
        query = query.filter(EducationalProgram.education_level == education_level)
    if department:
        query = query.filter(EducationalProgram.department == department)
    if max_price:
        query = query.filter(EducationalService.current_price <= float(max_price))
    service_list = query.order_by(EducationalProgram.name.asc()).all()
    regions = [item[0] for item in db.session.query(EducationalService.region).distinct().all()]
    formats = [item[0] for item in db.session.query(EducationalService.study_format).distinct().all()]
    levels = [item[0] for item in db.session.query(EducationalProgram.education_level).distinct().all()]
    departments = [item[0] for item in db.session.query(EducationalProgram.department).distinct().all()]
    return render_template(
        "services.html",
        services=service_list,
        regions=regions,
        formats=formats,
        levels=levels,
        departments=departments,
        selected_region=region,
        selected_format=study_format,
        selected_level=education_level,
        selected_department=department,
        max_price=max_price,
    )


@login_required
def service_card(service_id):
    service = EducationalService.query.get_or_404(service_id)
    price_history = PriceHistory.query.filter_by(service_id=service_id).order_by(PriceHistory.year.asc()).all()
    demand = DemandIndicator.query.filter_by(service_id=service_id).order_by(DemandIndicator.year.asc()).all()
    competitors = CompetitorPrice.query.filter_by(service_id=service_id).order_by(CompetitorPrice.year.asc()).all()
    forecasts = ForecastResult.query.filter_by(service_id=service_id).order_by(ForecastResult.created_at.desc()).limit(5).all()
    return render_template("service_card.html", service=service, price_history=price_history, demand=demand, competitors=competitors, forecasts=forecasts)


@login_required
@role_required("admin", "analyst")
def price_management():
    services_list = EducationalService.query.order_by(EducationalService.id.asc()).all()
    selected_service_id = request.args.get("service_id") or request.form.get("service_id")
    selected_service = None
    history = []
    competitors = []
    stats = {"mean": 0, "min": 0, "max": 0}
    if request.method == "POST":
        try:
            selected_service = EducationalService.query.get_or_404(int(selected_service_id))
            year = int(request.form.get("year"))
            price = float(request.form.get("price"))
            discount_percent = float(request.form.get("discount_percent", 0) or 0)
            student_count = int(request.form.get("student_count", 0) or 0)
            record = PriceHistory.query.filter_by(service_id=selected_service.id, year=year).first()
            if record is None:
                record = PriceHistory(service_id=selected_service.id, year=year)
                db.session.add(record)
            record.price = price
            record.discount_percent = discount_percent
            record.student_count = student_count
            selected_service.current_price = price
            db.session.commit()
            flash("Цена добавлена в историю и текущая стоимость обновлена.", "success")
            return redirect(url_for("price_management", service_id=selected_service.id))
        except Exception as exc:
            db.session.rollback()
            flash(str(exc), "danger")
    if selected_service_id:
        selected_service = EducationalService.query.get(int(selected_service_id))
    if selected_service is None and services_list:
        selected_service = services_list[0]
    if selected_service:
        history = PriceHistory.query.filter_by(service_id=selected_service.id).order_by(PriceHistory.year.asc()).all()
        competitors = CompetitorPrice.query.filter_by(service_id=selected_service.id).order_by(CompetitorPrice.year.asc()).all()
        prices = [item.price for item in history]
        if prices:
            stats = {
                "mean": round(sum(prices) / len(prices), 2),
                "min": round(min(prices), 2),
                "max": round(max(prices), 2),
            }
    return render_template(
        "price_management.html",
        services=services_list,
        selected_service=selected_service,
        history=history,
        competitors=competitors,
        stats=stats,
    )


@login_required
@role_required("admin", "analyst")
def upload_dataset():
    """Принять набор данных из файла или по ссылке и подготовить его для ML."""

    datasets = DatasetFile.query.order_by(DatasetFile.uploaded_at.desc()).all()
    preview_rows, preview_columns = [], []
    if request.method == "POST":
        try:
            dataset_url = request.form.get("dataset_url", "").strip()
            file = request.files.get("dataset")
            if dataset_url:
                saved_path, row_count = download_dataset_from_url(
                    dataset_url,
                    Config.UPLOAD_FOLDER,
                    max_bytes=Config.MAX_CONTENT_LENGTH,
                )
                original_name = dataset_url
            else:
                saved_path, row_count = save_uploaded_file(file, Config.UPLOAD_FOLDER)
                original_name = file.filename
            copy_dataset_to_processed(saved_path, Config.PROCESSED_DATASET_PATH)
            dataset = DatasetFile(
                filename=saved_path.name,
                original_name=original_name,
                row_count=row_count,
                status="загружен",
                description=request.form.get("description", ""),
            )
            db.session.add(dataset)
            db.session.commit()
            flash("Набор данных загружен и подготовлен для дальнейшего анализа.", "success")
            return redirect(url_for("upload_dataset"))
        except Exception as exc:
            db.session.rollback()
            flash(str(exc), "danger")
    try:
        preview_rows, preview_columns = dataset_preview(Config.PROCESSED_DATASET_PATH, limit=12)
    except Exception:
        preview_rows, preview_columns = [], []
    return render_template("upload_dataset.html", datasets=datasets, preview_rows=preview_rows, preview_columns=preview_columns)


@login_required
@role_required("admin", "analyst")
def delete_dataset(dataset_id):
    dataset = DatasetFile.query.get_or_404(dataset_id)
    upload_path = Path(Config.UPLOAD_FOLDER) / dataset.filename
    if upload_path.exists():
        upload_path.unlink()
    db.session.delete(dataset)
    db.session.commit()
    flash("Набор данных удален.", "success")
    return redirect(request.referrer or url_for("upload_dataset"))


@login_required
def price_analysis():
    try:
        summary = dataset_summary(Config.PROCESSED_DATASET_PATH)
        dynamics = price_dynamics(Config.PROCESSED_DATASET_PATH)
        rating = program_rating(Config.PROCESSED_DATASET_PATH)
        by_organization = category_analysis(
            Config.PROCESSED_DATASET_PATH,
            "organization",
        )
        by_program = category_analysis(Config.PROCESSED_DATASET_PATH, "program_name")
        competitors = competitor_analysis(Config.PROCESSED_DATASET_PATH)
        demand = demand_analysis(Config.PROCESSED_DATASET_PATH)
    except Exception as exc:
        summary, dynamics, rating, by_organization, by_program, competitors, demand = {}, [], [], [], [], [], []
        flash(f"Не удалось выполнить анализ набора данных: {exc}", "danger")
    return render_template(
        "price_analysis.html",
        summary=summary,
        dynamics=dynamics,
        rating=rating,
        by_organization=by_organization,
        by_program=by_program,
        competitors=competitors,
        demand=demand,
    )


@login_required
def forecast():
    """Рассчитать и сохранить прогноз выбранной пользователем моделью."""

    prediction = None
    services = EducationalService.query.order_by(EducationalService.id.asc()).all()
    models = available_models(Config.MODEL_DIR)
    default_model = selected_model_name(Config.MODEL_REGISTRY_PATH)
    selected_model = request.form.get("model_name", default_model)
    registry = load_model_registry(Config.MODEL_REGISTRY_PATH)
    metrics = registry.get("models", {}).get(selected_model)
    if request.method == "POST":
        try:
            model_path = model_path_for(selected_model, Config.MODEL_DIR)
            result = predict_price(model_path, request.form, model_name=selected_model)
            prediction = result
            record = ForecastResult(
                created_by=current_user.id,
                year=result["input"]["year"],
                organization=result["input"]["organization"],
                region="не используется моделью",
                education_level="не используется моделью",
                program_name=result["input"]["program_name"],
                study_format="не используется моделью",
                input_payload=json.dumps(result["input"], ensure_ascii=False),
                model_name=selected_model,
                predicted_price=result["predicted_price"],
                recommendation=result["recommendation"],
            )
            db.session.add(record)
            db.session.commit()
            flash("Прогноз успешно рассчитан и сохранен.", "success")
        except Exception as exc:
            db.session.rollback()
            flash(str(exc), "danger")
    return render_template(
        "forecast.html",
        prediction=prediction,
        services=services,
        models=models,
        selected_model=selected_model,
        model_display_names=MODEL_DISPLAY_NAMES,
        metrics=metrics,
    )


@login_required
def forecast_service(service_id):
    """Сформировать прогноз услуги автоматически выбранной лучшей моделью."""

    service = EducationalService.query.get_or_404(service_id)
    try:
        payload = prepare_forecast_payload_from_service(service)
        model_name = selected_model_name(Config.MODEL_REGISTRY_PATH)
        result = predict_price(
            model_path_for(model_name, Config.MODEL_DIR),
            payload,
            model_name=model_name,
        )
        forecast_record = ForecastResult(
            service_id=service.id,
            created_by=current_user.id,
            year=result["input"]["year"],
            organization=result["input"]["organization"],
            region="не используется моделью",
            education_level="не используется моделью",
            program_name=result["input"]["program_name"],
            study_format="не используется моделью",
            input_payload=json.dumps(result["input"], ensure_ascii=False),
            model_name=model_name,
            predicted_price=result["predicted_price"],
            recommendation=result["recommendation"],
        )
        db.session.add(forecast_record)
        db.session.commit()
        flash("Прогноз по выбранной услуге сформирован.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("service_card", service_id=service_id))


@login_required
@role_required("admin", "analyst")
def model_training():
    """Запустить единое сравнение трех моделей и показать полный реестр."""

    if request.method == "POST":
        try:
            train_all_models(
                Config.PROCESSED_DATASET_PATH,
                Config.MODEL_DIR,
                registry_path=Config.MODEL_REGISTRY_PATH,
                diagnostics_dir=Config.DIAGNOSTICS_DIR,
            )
            flash("Три модели обучены, сравнены и сохранены.", "success")
        except Exception as exc:
            flash(str(exc), "danger")
    registry = load_model_registry(Config.MODEL_REGISTRY_PATH)
    return render_template(
        "model_training.html",
        registry=registry,
        model_display_names=MODEL_DISPLAY_NAMES,
    )


@login_required
def model_diagnostic(filename):
    """Отдать сохраненный диагностический график из разрешенного каталога."""

    return send_from_directory(Config.DIAGNOSTICS_DIR, filename)


@login_required
def reports():
    forecasts = ForecastResult.query.order_by(ForecastResult.created_at.desc()).limit(50).all()
    report_files = ReportFile.query.order_by(ReportFile.created_at.desc()).limit(20).all()
    return render_template("reports.html", forecasts=forecasts, report_files=report_files)


@login_required
def export_xlsx():
    forecasts = ForecastResult.query.order_by(ForecastResult.created_at.desc()).limit(100).all()
    if not forecasts:
        flash("Нет сохраненных прогнозов для формирования отчета.", "warning")
        return redirect(url_for("reports"))
    path = export_forecasts_to_xlsx(Config.EXPORT_FOLDER, forecasts)
    report = ReportFile(created_by=current_user.id, filename=path.name, report_type="xlsx", file_path=str(path))
    db.session.add(report)
    db.session.commit()
    return send_file(path, as_attachment=True)


@login_required
def export_docx():
    forecasts = ForecastResult.query.order_by(ForecastResult.created_at.desc()).limit(100).all()
    if not forecasts:
        flash("Нет сохраненных прогнозов для формирования отчета.", "warning")
        return redirect(url_for("reports"))
    path = export_forecasts_to_docx(Config.EXPORT_FOLDER, forecasts)
    report = ReportFile(created_by=current_user.id, filename=path.name, report_type="docx", file_path=str(path))
    db.session.add(report)
    db.session.commit()
    return send_file(path, as_attachment=True)


@login_required
def export_dataset():
    try:
        summary = dataset_summary(Config.PROCESSED_DATASET_PATH)
        dynamics = price_dynamics(Config.PROCESSED_DATASET_PATH)
        rating = program_rating(Config.PROCESSED_DATASET_PATH)
        path = export_dataset_summary_to_xlsx(Config.EXPORT_FOLDER, summary, dynamics, rating)
        report = ReportFile(created_by=current_user.id, filename=path.name, report_type="dataset_xlsx", file_path=str(path))
        db.session.add(report)
        db.session.commit()
        return send_file(path, as_attachment=True)
    except Exception as exc:
        flash(str(exc), "danger")
        return redirect(url_for("price_analysis"))


@login_required
def export_analysis_docx():
    try:
        summary = dataset_summary(Config.PROCESSED_DATASET_PATH)
        dynamics = price_dynamics(Config.PROCESSED_DATASET_PATH)
        rating = program_rating(Config.PROCESSED_DATASET_PATH)
        by_organization = category_analysis(Config.PROCESSED_DATASET_PATH, "organization")
        competitors = competitor_analysis(Config.PROCESSED_DATASET_PATH)
        path = export_dataset_summary_to_docx(Config.EXPORT_FOLDER, summary, dynamics, rating, by_organization, competitors)
        report = ReportFile(created_by=current_user.id, filename=path.name, report_type="analysis_docx", file_path=str(path))
        db.session.add(report)
        db.session.commit()
        return send_file(path, as_attachment=True)
    except Exception as exc:
        flash(str(exc), "danger")
        return redirect(url_for("price_analysis"))


@login_required
def help_page():
    return render_template("help.html")


@login_required
def feedback():
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()
        email = request.form.get("email", current_user.email).strip()
        full_name = request.form.get("full_name", current_user.full_name).strip()
        if not subject or not message or not email or not full_name:
            flash("Заполните тему, сообщение, ФИО и email.", "warning")
        else:
            item = FeedbackMessage(
                created_by=current_user.id,
                full_name=full_name,
                email=email,
                subject=subject,
                message=message,
            )
            db.session.add(item)
            db.session.commit()
            flash("Сообщение отправлено администратору.", "success")
            return redirect(url_for("feedback"))
    return render_template("feedback.html")


@login_required
@role_required("admin")
def admin_panel():
    metrics = {
        "users": User.query.count(),
        "roles": Role.query.count(),
        "programs": EducationalProgram.query.count(),
        "services": EducationalService.query.count(),
        "datasets": DatasetFile.query.count(),
        "reports": ReportFile.query.count(),
        "feedback": FeedbackMessage.query.count(),
        "settings": SystemSetting.query.count(),
    }
    return render_template("admin_panel.html", metrics=metrics)


@login_required
@role_required("admin")
def admin_users():
    roles = Role.query.all()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role_id = request.form.get("role_id")
        if not username or not full_name or not email or not password or not role_id:
            flash("Заполните все поля пользователя.", "warning")
        elif User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Пользователь с такими данными уже существует.", "danger")
        else:
            user = User(username=username, full_name=full_name, email=email, password_hash=make_password_hash(password), role_id=int(role_id), is_active=True)
            db.session.add(user)
            db.session.commit()
            flash("Пользователь добавлен.", "success")
            return redirect(url_for("admin_users"))
    users = User.query.order_by(User.id.asc()).all()
    return render_template("admin_users.html", users=users, roles=roles)


@login_required
@role_required("admin")
def admin_toggle_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("Нельзя отключить собственную учетную запись.", "warning")
        return redirect(url_for("admin_users"))
    user.is_active = not user.is_active
    db.session.commit()
    flash("Статус пользователя изменен.", "success")
    return redirect(url_for("admin_users"))


@login_required
@role_required("admin")
def admin_roles():
    roles = Role.query.order_by(Role.id.asc()).all()
    users = User.query.order_by(User.id.asc()).all()
    if request.method == "POST":
        user = User.query.get_or_404(int(request.form.get("user_id")))
        role = Role.query.get_or_404(int(request.form.get("role_id")))
        user.role_id = role.id
        db.session.commit()
        flash("Роль пользователя обновлена.", "success")
        return redirect(url_for("admin_roles"))
    return render_template("admin_roles.html", roles=roles, users=users)


@login_required
@role_required("admin", "analyst")
def admin_programs():
    if request.method == "POST":
        program = EducationalProgram(
            name=request.form.get("name", "").strip(),
            education_level=request.form.get("education_level", "").strip(),
            department=request.form.get("department", "").strip(),
            duration_months=int(request.form.get("duration_months", 0) or 0),
            description=request.form.get("description", ""),
        )
        if not program.name or not program.education_level or not program.department or not program.duration_months:
            flash("Заполните обязательные поля программы.", "warning")
        else:
            db.session.add(program)
            db.session.commit()
            flash("Образовательная программа добавлена.", "success")
            return redirect(url_for("admin_programs"))
    programs = EducationalProgram.query.order_by(EducationalProgram.name.asc()).all()
    return render_template("admin_programs.html", programs=programs)


@login_required
@role_required("admin", "analyst")
def admin_edit_program(program_id):
    program = EducationalProgram.query.get_or_404(program_id)
    if request.method == "POST":
        try:
            program.name = request.form.get("name", "").strip()
            program.education_level = request.form.get("education_level", "").strip()
            program.department = request.form.get("department", "").strip()
            program.duration_months = int(request.form.get("duration_months", 0) or 0)
            program.description = request.form.get("description", "")
            if not program.name or not program.education_level or not program.department or not program.duration_months:
                flash("Заполните обязательные поля программы.", "warning")
            else:
                db.session.commit()
                flash("Программа обновлена.", "success")
                return redirect(url_for("admin_programs"))
        except Exception as exc:
            db.session.rollback()
            flash(str(exc), "danger")
    return render_template("admin_program_form.html", program=program)


@login_required
@role_required("admin")
def admin_delete_program(program_id):
    program = EducationalProgram.query.get_or_404(program_id)
    if program.services:
        flash("Нельзя удалить программу, к которой привязаны услуги.", "danger")
        return redirect(url_for("admin_programs"))
    db.session.delete(program)
    db.session.commit()
    flash("Программа удалена.", "success")
    return redirect(url_for("admin_programs"))


@login_required
@role_required("admin", "analyst")
def admin_services():
    programs = EducationalProgram.query.order_by(EducationalProgram.name.asc()).all()
    if request.method == "POST":
        try:
            service = EducationalService(
                program_id=int(request.form.get("program_id")),
                name=request.form.get("name", "").strip(),
                study_format=request.form.get("study_format", "").strip(),
                region=request.form.get("region", "").strip(),
                current_price=float(request.form.get("current_price", 0) or 0),
                is_active=True,
            )
            if not service.name or not service.study_format or not service.region or service.current_price <= 0:
                flash("Заполните обязательные поля услуги.", "warning")
            else:
                db.session.add(service)
                db.session.commit()
                flash("Образовательная услуга добавлена.", "success")
                return redirect(url_for("admin_services"))
        except Exception as exc:
            db.session.rollback()
            flash(str(exc), "danger")
    services = EducationalService.query.order_by(EducationalService.id.desc()).all()
    return render_template("admin_services.html", services=services, programs=programs)


@login_required
@role_required("admin", "analyst")
def admin_edit_service(service_id):
    service = EducationalService.query.get_or_404(service_id)
    programs = EducationalProgram.query.order_by(EducationalProgram.name.asc()).all()
    if request.method == "POST":
        try:
            service.program_id = int(request.form.get("program_id"))
            service.name = request.form.get("name", "").strip()
            service.study_format = request.form.get("study_format", "").strip()
            service.region = request.form.get("region", "").strip()
            service.current_price = float(request.form.get("current_price", 0) or 0)
            service.is_active = bool(request.form.get("is_active"))
            if not service.name or not service.study_format or not service.region or service.current_price <= 0:
                flash("Заполните обязательные поля услуги.", "warning")
            else:
                db.session.commit()
                flash("Услуга обновлена.", "success")
                return redirect(url_for("admin_services"))
        except Exception as exc:
            db.session.rollback()
            flash(str(exc), "danger")
    return render_template("admin_service_form.html", service=service, programs=programs)


@login_required
@role_required("admin")
def admin_delete_service(service_id):
    service = EducationalService.query.get_or_404(service_id)
    db.session.delete(service)
    db.session.commit()
    flash("Услуга удалена.", "success")
    return redirect(url_for("admin_services"))


@login_required
@role_required("admin")
def admin_datasets():
    datasets = DatasetFile.query.order_by(DatasetFile.uploaded_at.desc()).all()
    return render_template("admin_datasets.html", datasets=datasets)


@login_required
@role_required("admin")
def admin_reports():
    reports_list = ReportFile.query.order_by(ReportFile.created_at.desc()).all()
    return render_template("admin_reports.html", reports=reports_list)


@login_required
@role_required("admin")
def admin_delete_report(report_id):
    report = ReportFile.query.get_or_404(report_id)
    path = Path(report.file_path)
    if path.exists():
        path.unlink()
    db.session.delete(report)
    db.session.commit()
    flash("Отчет удален.", "success")
    return redirect(url_for("admin_reports"))


@login_required
@role_required("admin")
def admin_feedback():
    messages = FeedbackMessage.query.order_by(FeedbackMessage.created_at.desc()).all()
    return render_template("admin_feedback.html", messages=messages)


@login_required
@role_required("admin")
def admin_settings():
    if request.method == "POST":
        for setting in SystemSetting.query.all():
            value = request.form.get(f"setting_{setting.id}")
            if value is not None:
                setting.value = value
        db.session.commit()
        flash("Настройки сохранены.", "success")
        return redirect(url_for("admin_settings"))
    settings = SystemSetting.query.order_by(SystemSetting.key.asc()).all()
    return render_template("admin_settings.html", settings=settings)


@login_required
def api_service_history(service_id):
    history = [item.to_dict() for item in PriceHistory.query.filter_by(service_id=service_id).order_by(PriceHistory.year.asc()).all()]
    demand = [item.to_dict() for item in DemandIndicator.query.filter_by(service_id=service_id).order_by(DemandIndicator.year.asc()).all()]
    competitors = [item.to_dict() for item in CompetitorPrice.query.filter_by(service_id=service_id).order_by(CompetitorPrice.year.asc()).all()]
    return {"history": history, "demand": demand, "competitors": competitors}
