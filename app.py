from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import analyze_dataframe
from data_loader import load_dataframe
from llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMSettings, check_llm
from security import check_user_instruction


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_TITLE = "LLM-агент для анализа данных"
CHARTS_ROOT = Path("charts")

MODEL = DEFAULT_MODEL


st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("LLM-агент для анализа данных")
st.caption("Streamlit + GateLLM API + Google Gemini 2.5 Flash + безопасный Python tool")

DEFAULT_TIMEOUT = 90
DEFAULT_MAX_TOOL_CALLS = 3

with st.sidebar:
    st.header("GateLLM API")
    api_key = st.text_input(
        "API key",
        value=os.getenv("GATELLM_API_KEY", ""),
        type="password",
        help="Ключ храни в файле .env и не загружай его в GitHub.",
    )
    base_url = st.text_input(
        "Base URL",
        value=os.getenv("GATELLM_BASE_URL", DEFAULT_BASE_URL),
        help="Для GateLLM: https://gatellm.ru/v1",
    )
    allow_fallback = st.checkbox(
        "Базовый анализ при ошибке API",
        value=True,
        help="Если API не ответит, приложение покажет безопасный pandas-анализ.",
    )

    settings = LLMSettings(
        api_key=api_key,
        base_url=base_url,
        model=MODEL,
        timeout=DEFAULT_TIMEOUT,
    )

    if st.button("Проверить API"):
        ok, message = check_llm(settings)
        if ok:
            st.success(message)
        else:
            st.error(message)

uploaded_file = st.file_uploader("Файл с данными", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        loaded = load_dataframe(uploaded_file, uploaded_file.name)
        df = loaded.df
    except Exception as exc:
        st.error(f"Не удалось прочитать файл: {exc}")
        st.stop()

    st.subheader("Предпросмотр данных")
    st.write(f"Размер: **{df.shape[0]} строк × {df.shape[1]} столбцов**")

    info_cols = st.columns(4)
    info_cols[0].metric("Строк", df.shape[0])
    info_cols[1].metric("Столбцов", df.shape[1])
    info_cols[2].metric("Пропусков", int(df.isna().sum().sum()))
    info_cols[3].metric("Дубликатов", int(df.duplicated().sum()))

    with st.expander("Техническая информация о загрузке файла", expanded=False):
        st.json(
            {
                "file_type": loaded.info.file_type,
                "encoding": loaded.info.encoding,
                "separator": loaded.info.separator,
                "decimal": loaded.info.decimal,
            }
        )

    st.dataframe(df.head(30), use_container_width=True)

    instruction = st.text_area(
        "Вопрос к данным",
        height=130,
        placeholder=(
            "Например: Построй корреляции и объясни ограничения анализа."
        ),
    )

    run_button = st.button("Запустить анализ", type="primary")

    if run_button:
        sec = check_user_instruction(instruction)
        if not sec.ok:
            st.error(sec.message)
            st.stop()
        if not api_key.strip():
            st.error("Добавь GateLLM API key в боковой панели или в файл .env.")
            st.stop()

        run_id = uuid.uuid4().hex[:8]
        charts_dir = CHARTS_ROOT / run_id
        if charts_dir.exists():
            shutil.rmtree(charts_dir)
        charts_dir.mkdir(parents=True, exist_ok=True)

        status_box = st.empty()
        progress_bar = st.progress(0)
        step_counter = {"value": 0}

        def progress(message: str) -> None:
            status_box.info(message)
            step_counter["value"] = min(step_counter["value"] + 1, 4)
            progress_bar.progress(step_counter["value"] / 4)

        with st.spinner("Анализирую датасет..."):
            try:
                result = analyze_dataframe(
                    df=df,
                    user_instruction=instruction,
                    settings=settings,
                    charts_dir=charts_dir,
                    max_tool_calls=DEFAULT_MAX_TOOL_CALLS,
                    allow_fallback=allow_fallback,
                    progress=progress,
                )
            except Exception as exc:
                st.error(f"Анализ завершился ошибкой: {exc}")
                st.stop()

        progress_bar.progress(1.0)
        status_box.success(f"Готово за {result.elapsed_seconds:.1f} сек.")

        if result.used_fallback:
            st.warning("Показан базовый Python-анализ, потому что LLM API не смогла завершить агентный анализ.")

        st.subheader("Итоговый отчет")
        st.markdown(result.report)

        if result.charts:
            st.subheader("Графики")
            for chart_path in result.charts:
                path = Path(chart_path)
                if path.exists():
                    st.image(str(path), caption=path.name, use_container_width=True)

        with st.expander("Журнал Python tool calls", expanded=False):
            for idx, log in enumerate(result.tool_logs, start=1):
                st.markdown(f"### Tool call {idx}: {log.reason}")
                st.code(log.code, language="python")
                if log.ok:
                    st.success("Код выполнен")
                else:
                    st.error("Код завершился ошибкой")
                if log.stdout:
                    st.text(log.stdout)
                if log.error:
                    st.code(log.error, language="text")
else:
    st.subheader("Начало работы")
    st.markdown(
        "Выбери файл с данными, затем задай вопрос: например, "
        "«построй корреляции с диагнозом» или «найди факторы, связанные с целевой переменной»."
    )