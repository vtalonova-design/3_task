"""Надежная загрузка CSV/XLSX для Streamlit-приложения."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
import pandas as pd


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "latin1")
CSV_SEPARATORS = (";", ",", "\t", "|")


@dataclass(frozen=True)
class LoadInfo:
    file_type: str
    encoding: str | None = None
    separator: str | None = None
    decimal: str | None = None


@dataclass(frozen=True)
class LoadedData:
    df: pd.DataFrame
    info: LoadInfo


def _read_bytes(file: BinaryIO | bytes) -> bytes:
    if isinstance(file, bytes):
        return file
    data = file.read()
    try:
        file.seek(0)
    except Exception:
        pass
    return data


def _decode(data: bytes) -> tuple[str, str]:
    last_error: Exception | None = None
    for enc in CSV_ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise ValueError(f"Не удалось определить кодировку CSV: {last_error}")
    raise ValueError("Пустой или некорректный файл")


def _sniff_separator(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
        return dialect.delimiter
    except csv.Error:
        counts = {sep: sample.count(sep) for sep in CSV_SEPARATORS}
        return max(counts, key=counts.get)


def _guess_decimal(text: str, sep: str) -> str:
    # Если часто встречаются числа вида 12,34, а разделитель не запятая,
    # считаем десятичным разделителем запятую.
    if sep != "," and re.search(r"\d+,\d+", text):
        return ","
    return "."


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def _coerce_object_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        s = df[col].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        candidate = s.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        converted = pd.to_numeric(candidate, errors="coerce")
        non_empty = s.notna().sum()
        if non_empty and converted.notna().sum() / non_empty >= 0.85:
            df[col] = converted
    return df


def load_dataframe(uploaded_file: BinaryIO | bytes, filename: str) -> LoadedData:
    name = filename.lower()
    data = _read_bytes(uploaded_file)

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
        df = _coerce_object_numbers(_clean_column_names(df))
        return LoadedData(df=df, info=LoadInfo(file_type="excel"))

    if not name.endswith(".csv"):
        raise ValueError("Поддерживаются только CSV, XLSX и XLS файлы.")

    text, encoding = _decode(data)
    sep = _sniff_separator(text)
    decimal = _guess_decimal(text, sep)

    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal)

    # Если все равно получился один столбец, пробуем альтернативные разделители.
    if df.shape[1] == 1:
        best_df = df
        best_sep = sep
        for alt_sep in CSV_SEPARATORS:
            alt_df = pd.read_csv(io.StringIO(text), sep=alt_sep, decimal=_guess_decimal(text, alt_sep))
            if alt_df.shape[1] > best_df.shape[1]:
                best_df = alt_df
                best_sep = alt_sep
        df = best_df
        sep = best_sep
        decimal = _guess_decimal(text, sep)

    df = _coerce_object_numbers(_clean_column_names(df))
    return LoadedData(df=df, info=LoadInfo(file_type="csv", encoding=encoding, separator=sep, decimal=decimal))