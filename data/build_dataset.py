"""Сформировать датасет цен из мониторинга качества приема НИУ ВШЭ.

Автор: Горячевская Екатерина Николевна
Тема ВКР: «Анализ и прогнозирование ценообразования на образовательные услуги».

Скрипт загружает детальные таблицы платного приема за 2018–2024 годы. Каждая
строка таблицы описывает реальный вуз и укрупненную группу направлений. Значения
не размножаются и не генерируются случайно.
"""

from __future__ import annotations

import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "educational_services_prices.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "prepared_dataset.csv"

HSE_SOURCES = {
    2018: "https://ege.hse.ru/ege/rating/2018/77479782/all/",
    2019: "https://ege.hse.ru/rating/2019/81058609/all/",
    2020: "https://ege.hse.ru/rating/2020/84025368/all/",
    2021: "https://ege.hse.ru/ege/rating/2021/87901186/all/",
    2022: "https://ege.hse.ru/ege/rating/2022/91645099/all/",
    2023: "https://ege.hse.ru/ege/rating/2023/95405491/all/",
    2024: "https://ege.hse.ru/ege/rating/2024/98633979/all/",
}
CPI_SOURCE = (
    "https://api.worldbank.org/v2/country/RUS/indicator/"
    "FP.CPI.TOTL?format=json&per_page=100"
)
USER_AGENT = "muiv-costs-dataset-builder/2.0"
REQUEST_TIMEOUT = 90


def download(url: str) -> bytes:
    """Загрузить открытый источник и проверить HTTP-статус."""

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.content


def _find_column(frame: pd.DataFrame, *fragments: str) -> str:
    """Найти столбец, название которого содержит один из фрагментов."""

    for column in frame.columns:
        normalized = str(column).lower().replace("ё", "е")
        if any(fragment.lower().replace("ё", "е") in normalized for fragment in fragments):
            return column
    raise ValueError(f"Не найден столбец по фрагментам: {fragments}")


def _single_number(value) -> float | None:
    """Извлечь одно опубликованное число; диапазоны намеренно исключить."""

    if pd.isna(value):
        return None
    normalized = str(value).replace("\xa0", " ").replace(",", ".")
    numbers = re.findall(r"\d+(?:\.\d+)?", normalized)
    if len(numbers) != 1:
        return None
    return float(numbers[0])


def parse_hse_table(content: bytes, year: int, source_url: str) -> pd.DataFrame:
    """Привести детальную HTML-таблицу НИУ ВШЭ к единой схеме."""

    tables = pd.read_html(BytesIO(content))
    frame = max(tables, key=len)
    group_column = _find_column(frame, "укрупнен")
    organization_column = _find_column(frame, "вуз")
    score_column = _find_column(frame, "качество приема")
    students_column = _find_column(
        frame,
        "зачислено на платные",
        "сколько человек зачислено на платные",
    )
    price_column = _find_column(frame, "стоимость обучения")

    records = []
    for source_row, row in frame.iterrows():
        price = _single_number(row[price_column])
        score = _single_number(row[score_column])
        students = _single_number(row[students_column])
        organization = str(row[organization_column]).strip()
        program_name = str(row[group_column]).strip()
        if (
            price is None
            or score is None
            or students is None
            or students <= 0
            or organization in {"", "nan"}
            or program_name in {"", "nan"}
        ):
            continue
        # В таблицах цена опубликована в тысячах рублей, несмотря на сокращенный
        # заголовок некоторых архивных страниц.
        price_rubles = price * 1000 if price < 10_000 else price
        records.append(
            {
                "year": year,
                "organization": organization,
                "program_name": program_name,
                "student_count": int(students),
                "admission_score": round(score, 2),
                "final_price": round(price_rubles, 2),
                "source_title": "НИУ ВШЭ — Мониторинг качества приема в вузы",
                "source_url": source_url,
                "source_row": int(source_row) + 1,
            }
        )
    return pd.DataFrame(records)


def load_cpi() -> dict[int, float]:
    """Загрузить ИПЦ РФ и перебазировать значения к 2018 году."""

    response = requests.get(
        CPI_SOURCE,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    values = {
        int(item["date"]): float(item["value"])
        for item in response.json()[1]
        if item["value"] is not None
    }
    base = values[2018]
    return {year: round(value / base * 100, 2) for year, value in values.items()}


def _aggregate_organization_program_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Оставить одно реальное наблюдение на вуз, направление и год."""

    frame["weighted_score"] = frame["admission_score"] * frame["student_count"]
    grouped = (
        frame.groupby(
            ["year", "organization", "program_name", "source_title", "source_url"],
            as_index=False,
        )
        .agg(
            student_count=("student_count", "sum"),
            weighted_score=("weighted_score", "sum"),
            final_price=("final_price", "median"),
            source_row=("source_row", "min"),
        )
    )
    grouped["admission_score"] = (
        grouped["weighted_score"] / grouped["student_count"]
    ).round(2)
    return grouped.drop(columns="weighted_score")


def _add_previous_price(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавить последнюю опубликованную цену того же вуза и направления."""

    frame = frame.sort_values(["organization", "program_name", "year"])
    history_group = frame.groupby(["organization", "program_name"])
    frame["base_price"] = history_group["final_price"].shift(1)
    frame["base_price_year"] = history_group["year"].shift(1)
    frame["base_price_source_url"] = history_group["source_url"].shift(1)
    frame["base_price_source_row"] = history_group["source_row"].shift(1)
    return frame


def _add_competitor_price(frame: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать медиану цен других вузов без текущей целевой строки."""

    frame = frame.copy()
    competitor = pd.Series(index=frame.index, dtype=float)
    for _, indexes in frame.groupby(["year", "program_name"]).groups.items():
        indexes = list(indexes)
        if len(indexes) < 2:
            continue
        prices = frame.loc[indexes, "final_price"].to_numpy(dtype=float)
        for position, index in enumerate(indexes):
            competitor.loc[index] = float(np.median(np.delete(prices, position)))
    frame["competitor_price"] = competitor.round(2)
    return frame


def build_dataset() -> pd.DataFrame:
    """Собрать реальный временной набор и удалить строки без истории/конкурентов."""

    yearly_frames = [
        parse_hse_table(download(url), year, url)
        for year, url in HSE_SOURCES.items()
    ]
    frame = pd.concat(yearly_frames, ignore_index=True)
    frame = _aggregate_organization_program_rows(frame)
    frame = _add_previous_price(frame)
    frame["salary_index"] = frame["year"].map(load_cpi())
    frame = frame.dropna(subset=["base_price", "salary_index"])
    frame = _add_competitor_price(frame)
    frame = frame.dropna(subset=["competitor_price"])
    frame["base_price_year"] = frame["base_price_year"].astype(int)
    frame = frame[
        (frame["final_price"] > 0)
        & (frame["base_price"] > 0)
        & (frame["competitor_price"] > 0)
    ]

    columns = [
        "year",
        "organization",
        "program_name",
        "base_price",
        "competitor_price",
        "student_count",
        "admission_score",
        "salary_index",
        "final_price",
        "base_price_year",
        "base_price_source_url",
        "base_price_source_row",
        "source_title",
        "source_url",
        "source_row",
    ]
    return frame[columns].sort_values(
        ["year", "organization", "program_name"]
    ).reset_index(drop=True)


def main() -> None:
    """Сохранить исходный набор с происхождением и рабочую ML-версию."""

    dataset = build_dataset()
    if len(dataset) < 1_000:
        raise RuntimeError(
            f"После проверок осталось только {len(dataset)} строк; требуется минимум 1000"
        )
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")

    sys.path.insert(0, str(ROOT))
    from app.ml_service import normalize_dataset

    normalize_dataset(dataset).to_csv(
        PROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Сформировано реальных строк: {len(dataset)}")
    print(f"Вузов: {dataset['organization'].nunique()}")
    print(f"Направлений: {dataset['program_name'].nunique()}")
    print(f"Период: {dataset['year'].min()}–{dataset['year'].max()}")


if __name__ == "__main__":
    main()
