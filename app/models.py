from datetime import datetime
from flask_login import UserMixin
from app import db


class Role(db.Model):
    __tablename__ = "roles"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    title = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    users = db.relationship("User", back_populates="role")

    def __repr__(self):
        return f"<Role {self.name}>"

    def to_dict(self):
        return {"id": self.id, "name": self.name, "title": self.title}


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    full_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey("roles.id"), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime)

    role = db.relationship("Role", back_populates="users")
    forecasts = db.relationship("ForecastResult", back_populates="created_by_user")
    reports = db.relationship("ReportFile", back_populates="created_by_user")
    feedback_messages = db.relationship("FeedbackMessage", back_populates="created_by_user")

    def __repr__(self):
        return f"<User {self.username}>"

    def get_id(self):
        return str(self.id)

    @property
    def is_admin(self):
        return self.role and self.role.name == "admin"

    @property
    def is_analyst(self):
        return self.role and self.role.name in ("admin", "analyst")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role.title if self.role else None,
            "is_active": self.is_active,
        }


class EducationalProgram(db.Model):
    __tablename__ = "educational_programs"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    education_level = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(200), nullable=False)
    duration_months = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    services = db.relationship("EducationalService", back_populates="program", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<EducationalProgram {self.name}>"

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "education_level": self.education_level,
            "department": self.department,
            "duration_months": self.duration_months,
        }


class EducationalService(db.Model):
    __tablename__ = "educational_services"

    id = db.Column(db.Integer, primary_key=True)
    program_id = db.Column(db.Integer, db.ForeignKey("educational_programs.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    study_format = db.Column(db.String(80), nullable=False)
    region = db.Column(db.String(120), nullable=False)
    current_price = db.Column(db.Float, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    program = db.relationship("EducationalProgram", back_populates="services")
    price_history = db.relationship("PriceHistory", back_populates="service", cascade="all, delete-orphan")
    demand_indicators = db.relationship("DemandIndicator", back_populates="service", cascade="all, delete-orphan")
    competitor_prices = db.relationship("CompetitorPrice", back_populates="service", cascade="all, delete-orphan")
    forecasts = db.relationship("ForecastResult", back_populates="service")

    def __repr__(self):
        return f"<EducationalService {self.name}>"

    @property
    def full_title(self):
        program_name = self.program.name if self.program else "Программа"
        return f"{program_name}: {self.name}, {self.study_format}, {self.region}"

    def last_price(self):
        record = PriceHistory.query.filter_by(service_id=self.id).order_by(PriceHistory.year.desc()).first()
        return record.price if record else self.current_price

    def to_dict(self):
        return {
            "id": self.id,
            "program": self.program.name if self.program else None,
            "name": self.name,
            "study_format": self.study_format,
            "region": self.region,
            "current_price": self.current_price,
            "is_active": self.is_active,
        }


class PriceHistory(db.Model):
    __tablename__ = "price_history"

    id = db.Column(db.Integer, primary_key=True)
    service_id = db.Column(db.Integer, db.ForeignKey("educational_services.id"), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    discount_percent = db.Column(db.Float, default=0)
    student_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    service = db.relationship("EducationalService", back_populates="price_history")

    __table_args__ = (db.UniqueConstraint("service_id", "year", name="uq_price_history_service_year"),)

    def to_dict(self):
        return {
            "id": self.id,
            "service_id": self.service_id,
            "year": self.year,
            "price": self.price,
            "discount_percent": self.discount_percent,
            "student_count": self.student_count,
        }


class DemandIndicator(db.Model):
    __tablename__ = "demand_indicators"

    id = db.Column(db.Integer, primary_key=True)
    service_id = db.Column(db.Integer, db.ForeignKey("educational_services.id"), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    demand_index = db.Column(db.Float, nullable=False)
    salary_index = db.Column(db.Float, nullable=False)
    advertising_spend = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    service = db.relationship("EducationalService", back_populates="demand_indicators")

    __table_args__ = (db.UniqueConstraint("service_id", "year", name="uq_demand_service_year"),)

    def to_dict(self):
        return {
            "id": self.id,
            "service_id": self.service_id,
            "year": self.year,
            "demand_index": self.demand_index,
            "salary_index": self.salary_index,
            "advertising_spend": self.advertising_spend,
        }


class CompetitorPrice(db.Model):
    __tablename__ = "competitor_prices"

    id = db.Column(db.Integer, primary_key=True)
    service_id = db.Column(db.Integer, db.ForeignKey("educational_services.id"), nullable=False)
    competitor_name = db.Column(db.String(200), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    source = db.Column(db.String(250), default="Не указан")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    service = db.relationship("EducationalService", back_populates="competitor_prices")

    def to_dict(self):
        return {
            "id": self.id,
            "service_id": self.service_id,
            "competitor_name": self.competitor_name,
            "year": self.year,
            "price": self.price,
            "source": self.source,
        }


class DatasetFile(db.Model):
    __tablename__ = "dataset_files"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(80), default="загружен")
    description = db.Column(db.Text)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "original_name": self.original_name,
            "row_count": self.row_count,
            "status": self.status,
            "uploaded_at": self.uploaded_at.strftime("%d.%m.%Y %H:%M") if self.uploaded_at else None,
        }


class ForecastResult(db.Model):
    __tablename__ = "forecast_results"

    id = db.Column(db.Integer, primary_key=True)
    service_id = db.Column(db.Integer, db.ForeignKey("educational_services.id"), nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    year = db.Column(db.Integer, nullable=False)
    region = db.Column(db.String(120), nullable=False)
    education_level = db.Column(db.String(120), nullable=False)
    program_name = db.Column(db.String(200), nullable=False)
    study_format = db.Column(db.String(80), nullable=False)
    input_payload = db.Column(db.Text)
    predicted_price = db.Column(db.Float, nullable=False)
    recommendation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    service = db.relationship("EducationalService", back_populates="forecasts")
    created_by_user = db.relationship("User", back_populates="forecasts")

    def to_dict(self):
        return {
            "id": self.id,
            "year": self.year,
            "region": self.region,
            "education_level": self.education_level,
            "program_name": self.program_name,
            "study_format": self.study_format,
            "predicted_price": self.predicted_price,
            "recommendation": self.recommendation,
        }


class ReportFile(db.Model):
    __tablename__ = "report_files"

    id = db.Column(db.Integer, primary_key=True)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    report_type = db.Column(db.String(80), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    created_by_user = db.relationship("User", back_populates="reports")

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "report_type": self.report_type,
            "file_path": self.file_path,
            "created_at": self.created_at.strftime("%d.%m.%Y %H:%M") if self.created_at else None,
        }


class SystemSetting(db.Model):
    __tablename__ = "system_settings"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(120), unique=True, nullable=False)
    value = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "description": self.description,
        }


class FeedbackMessage(db.Model):
    __tablename__ = "feedback_messages"

    id = db.Column(db.Integer, primary_key=True)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    full_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(80), default="новое")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    created_by_user = db.relationship("User", back_populates="feedback_messages")

    def to_dict(self):
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "subject": self.subject,
            "message": self.message,
            "status": self.status,
            "created_at": self.created_at.strftime("%d.%m.%Y %H:%M") if self.created_at else None,
        }
