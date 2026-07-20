"""Сформировать датасет цен из подтверждаемых открытых источников.

Автор: Горячевская Екатерина Николевна
Тема ВКР: «Анализ и прогнозирование ценообразования на образовательные услуги».

Скрипт загружает опубликованные Роспатентом версии набора платных услуг РГАИС,
добавляет открытые сведения РУДН о численности студентов по направлениям и
индекс потребительских цен Всемирного банка. Случайные значения не создаются.
Все производные признаки описаны в ``data/SOURCES.md``.
"""

from __future__ import annotations

import csv
import re
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "educational_services_prices.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "prepared_dataset.csv"

PRICE_SOURCES = {
    2017: "https://rospatent.gov.ru/opendata/7730176088-paidservices/data-20171019-structure-20171019.csv",
    2021: "https://rospatent.gov.ru/opendata/7730176088-paidservices/data-20210629-structure-20210629.csv",
    2022: "https://rospatent.gov.ru/opendata/7730176088-paidservices/data-20220905-structure-20210629.csv",
    2023: "https://rospatent.gov.ru/opendata/7730176088-paidservices/data-20231004-structure-20210629.csv",
    2024: "https://rospatent.gov.ru/opendata/7730176088-paidservices/data-20240902-structure-20210629.csv",
}
STUDENT_SOURCE = "https://opendata.rudn.ru/file/293/data_01072026_structure_01072026.csv"
CPI_SOURCE = "https://api.worldbank.org/v2/country/RUS/indicator/FP.CPI.TOTL?format=json&per_page=100"

USER_AGENT = "muiv-costs-dataset-builder/1.0"
REQUEST_TIMEOUT = 60


def download(url: str) -> bytes:
    """Загрузить источник и проверить HTTP-статус ответа."""

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.content


def _parse_source_line(line: str) -> tuple[str, str, str, str] | None:
    """Разобрать обычную или исторически некорректно экранированную CSV-строку."""

    fields = next(csv.reader([line]))
    if len(fields) >= 4:
        return tuple(fields[:4])

    inner = fields[0]
    # В версиях 2021–2024 каждая строка дополнительно обернута в кавычки.
    while inner.endswith(',""'):
        inner = inner[:-3]
    parts = inner.rsplit('","', 3)
    if len(parts) != 4:
        return None
    name, education_type, cost, note = parts
    return (
        name.strip().strip('"'),
        education_type.strip().strip('"'),
        cost.strip().strip('"'),
        note.strip().strip('"'),
    )


def parse_price_source(content: bytes, snapshot_year: int, source_url: str) -> list[dict]:
    """Извлечь отдельную запись для каждой формы обучения и опубликованной цены."""

    text = content.decode("cp1251")
    rows = []
    for source_row, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lower().startswith("name,types,cost,note"):
            continue
        parsed = _parse_source_line(line)
        if not parsed:
            continue
        name, education_type, cost_text, note = parsed
        education_level = normalize_education_level(education_type)
        program_name = normalize_program_name(name)
        direction_code = extract_direction_code(name)
        for study_format, price in parse_prices(cost_text):
            rows.append(
                {
                    "year": snapshot_year,
                    "region": "Москва",
                    "education_level": education_level,
                    "program_name": program_name,
                    "study_format": study_format,
                    "duration_months": duration_for_level(education_level),
                    "final_price": price,
                    "discount_percent": 0.0,
                    "advertising_spend": 0.0,
                    "direction_code": direction_code,
                    "source_title": "Открытые данные Роспатента: платные образовательные услуги РГАИС",
                    "source_url": source_url,
                    "source_row": source_row,
                    "source_note": note,
                }
            )
    return rows


def parse_prices(cost_text: str) -> list[tuple[str, float]]:
    """Преобразовать текстовый перечень цен в пары «форма — цена»."""

    prepared = re.sub(
        r"\s+(?=(?:очно-заочная|заочная|очная|с применением дистанционных|полный курс))",
        "; ",
        cost_text.lower(),
    )
    results = []
    for segment in prepared.split(";"):
        numbers = re.findall(r"\d[\d ]*(?:,\d+)?", segment)
        if not numbers:
            continue
        price = float(numbers[-1].replace(" ", "").replace(",", "."))
        if price < 1_000:
            continue
        if "дистанцион" in segment:
            study_format = "дистанционная"
        elif "очно-заоч" in segment or "очная/очно-заочная" in segment:
            study_format = "очно-заочная"
        elif "заоч" in segment:
            study_format = "заочная"
        elif "очная" in segment:
            study_format = "очная"
        else:
            study_format = "не указано"
        results.append((study_format, price))
    return results


def extract_direction_code(name: str) -> str:
    """Извлечь код направления вида 38.03.02, если он опубликован."""

    match = re.search(r"\b\d{2}\.\d{2}\.\d{2}\b", name)
    return match.group(0) if match else ""


def normalize_program_name(name: str) -> str:
    """Очистить название услуги от технических кавычек и лишних пробелов."""

    cleaned = re.sub(r"\s+", " ", name.replace('"', "")).strip(" ,.")
    return cleaned[:240]


def normalize_education_level(value: str) -> str:
    """Привести опубликованный тип программы к категориям приложения."""

    lowered = value.lower()
    if "бакалавр" in lowered:
        return "бакалавриат"
    if "магистр" in lowered:
        return "магистратура"
    if "специалитет" in lowered:
        return "специалитет"
    if "аспиран" in lowered:
        return "аспирантура"
    if "высшее" in lowered:
        return "высшее образование"
    return "дополнительное образование"


def duration_for_level(level: str) -> int:
    """Вернуть нормативную продолжительность программы в месяцах."""

    return {
        "бакалавриат": 48,
        "магистратура": 24,
        "специалитет": 60,
        "аспирантура": 36,
        "высшее образование": 48,
        "дополнительное образование": 6,
    }[level]


def load_student_counts() -> dict[str, int]:
    """Получить численность студентов РУДН по кодам направлений."""

    frame = pd.read_csv(BytesIO(download(STUDENT_SOURCE)), sep=";", encoding="cp1251")
    code_column = "Шифр направления подготовки"
    course_columns = [column for column in frame.columns if "курс" in column]
    for column in course_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["student_count"] = frame[course_columns].sum(axis=1)
    return (
        frame.groupby(code_column)["student_count"]
        .sum()
        .round()
        .astype(int)
        .to_dict()
    )


def load_cpi() -> dict[int, float]:
    """Загрузить индекс потребительских цен РФ и перебазировать к 2017 году."""

    response = requests.get(
        CPI_SOURCE,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    records = response.json()[1]
    values = {
        int(item["date"]): float(item["value"])
        for item in records
        if item["value"] is not None
    }
    base = values[2017]
    return {year: round(value / base * 100, 2) for year, value in values.items()}


def build_dataset() -> pd.DataFrame:
    """Собрать, обогатить и вернуть итоговую таблицу без случайной генерации."""

    records = []
    for year, url in PRICE_SOURCES.items():
        records.extend(parse_price_source(download(url), year, url))
    if not records:
        raise RuntimeError("Из источников не удалось извлечь цены")

    frame = pd.DataFrame(records)
    frame = frame.drop_duplicates(
        subset=["year", "program_name", "study_format", "final_price", "source_url"]
    )
    student_counts = load_student_counts()
    known_counts = [value for value in student_counts.values() if value > 0]
    fallback_students = int(pd.Series(known_counts).median())
    frame["student_count"] = frame["direction_code"].map(student_counts).fillna(fallback_students).astype(int)

    min_students = frame["student_count"].min()
    max_students = frame["student_count"].max()
    if min_students == max_students:
        frame["demand_index"] = 50.0
    else:
        frame["demand_index"] = (
            20
            + (frame["student_count"] - min_students)
            / (max_students - min_students)
            * 80
        ).round(2)

    cpi = load_cpi()
    frame["salary_index"] = frame["year"].map(cpi)

    peer_group = ["year", "education_level", "study_format"]
    frame["competitor_price"] = frame.groupby(peer_group)["final_price"].transform("median")
    frame["competitor_price"] = frame["competitor_price"].fillna(
        frame.groupby("year")["final_price"].transform("median")
    )

    frame = frame.sort_values(["program_name", "study_format", "year", "final_price"])
    frame["base_price"] = frame.groupby(["program_name", "study_format"])["final_price"].shift(1)
    frame["base_price"] = frame["base_price"].fillna(frame["competitor_price"])

    ordered_columns = [
        "year",
        "region",
        "education_level",
        "program_name",
        "study_format",
        "duration_months",
        "base_price",
        "competitor_price",
        "demand_index",
        "salary_index",
        "advertising_spend",
        "discount_percent",
        "student_count",
        "final_price",
        "direction_code",
        "source_title",
        "source_url",
        "source_row",
        "source_note",
    ]
    return frame[ordered_columns].reset_index(drop=True)


def main() -> None:
    """Сохранить исходную таблицу с происхождением и очищенную ML-версию."""

    dataset = build_dataset()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")

    sys.path.insert(0, str(ROOT))
    from app.ml_service import normalize_dataset

    normalize_dataset(dataset).to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    print(f"Сформировано строк: {len(dataset)}")
    print(f"Исходный набор: {RAW_PATH}")
    print(f"Набор для ML: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
