"""Python-инструмент агента для анализа DataFrame."""
from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from security import check_python_code


SAFE_BUILTINS = {
    "__import__": __import__,  
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "object": object,
    "type": type,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def dataframe_overview(df: pd.DataFrame, max_rows: int = 5) -> str:
    head = df.head(max_rows).to_dict(orient="records")
    missing = df.isna().sum().sort_values(ascending=False).head(20)
    overview = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "top_missing_columns": {str(k): int(v) for k, v in missing.items()},
        "sample_rows": head,
    }
    return json.dumps(overview, ensure_ascii=False, indent=2, default=str)


def save_chart(fig: Any, charts_dir: str | Path, name: str = "chart") -> str:
    """Сохраняет matplotlib-график и возвращает путь к файлу."""
    charts_path = Path(charts_dir)
    charts_path.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60] or "chart"
    out_path = charts_path / f"{safe_name}_{len(list(charts_path.glob('*.png'))) + 1}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def run_python_tool(df: pd.DataFrame, code: str, charts_dir: str | Path) -> dict[str, Any]:
    security = check_python_code(code)
    if not security.ok:
        return {
            "ok": False,
            "stdout": "",
            "error": security.message,
            "charts": [],
            "code": code,
        }

    before_charts = set(Path(charts_dir).glob("*.png")) if Path(charts_dir).exists() else set()
    stdout = io.StringIO()

    exec_env: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "charts_dir": str(charts_dir),
        "save_chart": lambda fig, name="chart": save_chart(fig, charts_dir, name),
    }

    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<llm_tool_code>", "exec"), exec_env, exec_env)
        after_charts = set(Path(charts_dir).glob("*.png")) if Path(charts_dir).exists() else set()
        new_charts = sorted(str(p) for p in after_charts - before_charts)
        return {
            "ok": True,
            "stdout": stdout.getvalue().strip(),
            "error": "",
            "charts": new_charts,
            "code": code,
        }
    except Exception:
        return {
            "ok": False,
            "stdout": stdout.getvalue().strip(),
            "error": traceback.format_exc(limit=5),
            "charts": [],
            "code": code,
        }


def build_auto_chart_code(user_instruction: str) -> str:
    question_literal = repr(user_instruction)
    return f'''
print("## Обязательный график")
question = {question_literal}.lower()
numeric_df = df.select_dtypes(include=[np.number]).copy()

possible_targets = []
target_keywords = ["target", "label", "diagnosis", "диагноз", "outcome", "result", "copd", "diabetes", "class"]
for col in df.columns:
    low = str(col).lower()
    if low in question or any(key in low for key in target_keywords):
        possible_targets.append(col)

if possible_targets:
    target = possible_targets[0]
    print("Целевая колонка:", target)
    y_raw = df[target]
    if y_raw.dtype == object:
        s = y_raw.astype(str).str.strip().str.lower()
        y = s.map({{"да": 1, "yes": 1, "true": 1, "1": 1, "нет": 0, "no": 0, "false": 0, "0": 0}})
        if y.notna().mean() < 0.50:
            codes, uniques = pd.factorize(s)
            y = pd.Series(codes, index=df.index).replace(-1, np.nan)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    feature_df = numeric_df.drop(columns=[target], errors="ignore").copy()
    for col in df.columns:
        if col == target or col in feature_df.columns:
            continue
        if df[col].dtype != object:
            continue
        s = df[col].astype(str).str.strip().str.lower()
        mapped = s.map({{"да": 1, "yes": 1, "true": 1, "1": 1, "нет": 0, "no": 0, "false": 0, "0": 0, "м": 1, "male": 1, "ж": 0, "female": 0}})
        if mapped.notna().mean() >= 0.70:
            feature_df[str(col)] = mapped

    if len(feature_df.columns) > 0 and y.notna().sum() >= 3:
        corr = feature_df.corrwith(y).dropna()
        corr = corr.loc[corr.abs().sort_values(ascending=False).index]
        print("Корреляции с целевой переменной:")
        print(corr.head(15).round(3).to_string())
        if len(corr) > 0:
            top = corr.head(12).sort_values()
            fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
            top.plot(kind="barh", ax=ax)
            ax.set_title(f"Корреляции с {{target}}")
            ax.set_xlabel("Корреляция")
            path = save_chart(fig, "target_correlations")
            print("График корреляций сохранен:", path)
    else:
        print("Недостаточно числовых признаков для корреляционного графика.")

elif len(numeric_df.columns) >= 2:
    corr = numeric_df.corr(numeric_only=True).round(2)
    print("Корреляционная матрица числовых признаков:")
    print(corr.to_string())
    fig, ax = plt.subplots(figsize=(min(10, len(corr.columns) * 0.7 + 3), min(8, len(corr.columns) * 0.6 + 3)))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([str(c) for c in corr.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(c) for c in corr.index])
    fig.colorbar(im, ax=ax)
    ax.set_title("Корреляционная матрица")
    path = save_chart(fig, "correlation_matrix")
    print("График корреляционной матрицы сохранен:", path)

else:
    categorical_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique(dropna=True) > 1]
    if categorical_cols:
        col = categorical_cols[0]
        counts = df[col].astype(str).value_counts().head(15)
        print("Распределение категорий в колонке:", col)
        print(counts.to_string())
        fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.35)))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Топ категорий: {{col}}")
        ax.set_xlabel("Количество")
        path = save_chart(fig, "category_distribution")
        print("График распределения категорий сохранен:", path)
    else:
        print("Не нашлось подходящих колонок для графика.")
'''.strip()


def build_basic_eda_code(user_instruction: str) -> str:
    """Запасной Python-анализ, если API недоступен."""
    question_literal = repr(user_instruction)
    auto_chart_code = build_auto_chart_code(user_instruction)
    return f'''
print("## Общая информация")
print("Размер датасета:", df.shape)
print("\\nТипы данных:")
print(df.dtypes.to_string())
print("\\nПропуски по колонкам:")
print(df.isna().sum().sort_values(ascending=False).to_string())
print("\\nДубликаты:", int(df.duplicated().sum()))

numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 0:
    print("\\n## Описательная статистика")
    print(numeric_df.describe().T.round(3).to_string())

# Попытка найти целевую колонку из вопроса или по популярным названиям.
question = {question_literal}.lower()
possible_targets = []
for col in df.columns:
    low = str(col).lower()
    if low in question or any(key in low for key in ["target", "label", "diagnosis", "диагноз", "outcome", "result", "copd", "diabetes", "class"]):
        possible_targets.append(col)

if len(possible_targets) > 0:
    target = possible_targets[0]
    print(f"\\n## Анализ целевой переменной: {{target}}")
    y_raw = df[target]
    if y_raw.dtype == object:
        y = y_raw.astype(str).str.strip().str.lower().map({{"да": 1, "yes": 1, "true": 1, "1": 1, "нет": 0, "no": 0, "false": 0, "0": 0, "м": 1, "male": 1, "ж": 0, "female": 0}})
        if y.notna().mean() < 0.5:
            codes, uniques = pd.factorize(y_raw.astype(str))
            y = pd.Series(codes, index=df.index).replace(-1, np.nan)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    if y.notna().sum() > 2:
        feature_df = numeric_df.drop(columns=[target], errors="ignore").copy()
        for col in df.columns:
            if col == target or col in feature_df.columns:
                continue
            if df[col].dtype != object:
                continue
            s = df[col].astype(str).str.strip().str.lower()
            mapped = s.map({{"да": 1, "yes": 1, "true": 1, "1": 1, "нет": 0, "no": 0, "false": 0, "0": 0, "м": 1, "male": 1, "ж": 0, "female": 0}})
            if mapped.notna().mean() >= 0.7:
                feature_df[str(col)] = mapped

        if len(feature_df.columns) > 0:
            corr = feature_df.corrwith(y).dropna()
            corr = corr.loc[corr.abs().sort_values(ascending=False).index]
            print("\\nКорреляции признаков с целевой переменной:")
            print(corr.head(15).round(3).to_string())
            if len(corr) > 0:
                top = corr.head(12).sort_values()
                fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
                top.plot(kind="barh", ax=ax)
                ax.set_title(f"Корреляции с {{target}}")
                ax.set_xlabel("Корреляция")
                path = save_chart(fig, "target_correlations")
                print("\\nГрафик сохранен:", path)

print("\\n")
{auto_chart_code}
'''.strip()