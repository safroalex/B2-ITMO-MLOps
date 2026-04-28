"""
ЭТАП 5: Streamlit UI — интерфейс для sentiment classification

UI обращается к ClearML Serving endpoint через HTTP.
Модель НЕ загружается напрямую — только HTTP-запросы к serving.

Запуск:
    streamlit run app.py

Serving endpoint по умолчанию: http://localhost:8082/serve/sentiment
Можно переопределить через переменную окружения:
    export SERVING_ENDPOINT=http://localhost:8082/serve/sentiment
"""

import os
import time

import requests
import streamlit as st

DEFAULT_ENDPOINT = os.getenv(
    "SERVING_ENDPOINT", "http://localhost:8082/serve/sentiment"
)

st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 IMDB Sentiment Classifier")
st.caption("Определение тональности отзыва (positive / negative) через ClearML Serving")

# --- Боковая панель ---
with st.sidebar:
    st.header("⚙️ Настройки")
    endpoint_url = st.text_input(
        "Serving Endpoint URL",
        value=DEFAULT_ENDPOINT,
        help="URL ClearML Serving inference endpoint",
    )
    timeout_sec = st.slider("Timeout (сек)", min_value=1, max_value=30, value=10)

    st.divider()
    st.markdown("**Примеры запросов:**")
    examples = [
        "This movie was absolutely fantastic! Best film I've seen in years.",
        "Terrible movie. Complete waste of time. Would not recommend.",
        "An average film with some good moments but mostly forgettable.",
    ]
    for ex in examples:
        if st.button(ex[:50] + "…", use_container_width=True):
            st.session_state["input_text"] = ex

# --- Основная форма ---
input_text = st.text_area(
    "Введи текст отзыва:",
    value=st.session_state.get("input_text", ""),
    height=150,
    placeholder="Enter your movie review here...",
    key="text_area",
)

col1, col2 = st.columns([1, 4])
predict_clicked = col1.button("🔮 Predict", type="primary", use_container_width=True)

if predict_clicked:
    if not input_text.strip():
        st.warning("Введи текст для классификации.")
    else:
        with st.spinner("Запрашиваем endpoint..."):
            start_time = time.perf_counter()
            try:
                response = requests.post(
                    endpoint_url,
                    json={"text": input_text.strip()},
                    timeout=timeout_sec,
                    headers={"Content-Type": "application/json"},
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code == 200:
                    result = response.json()
                    label = result.get("label", "unknown")

                    if label == "positive":
                        st.success(f"**Результат: {label.upper()}** 👍")
                    elif label == "negative":
                        st.error(f"**Результат: {label.upper()}** 👎")
                    else:
                        st.info(f"**Результат: {label.upper()}**")

                    col_a, col_b = st.columns(2)
                    col_a.metric("Label", label.capitalize())
                    col_b.metric("Latency", f"{latency_ms:.1f} ms")

                    with st.expander("Raw response"):
                        st.json(result)

                else:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    st.error(
                        f"Ошибка сервера: HTTP {response.status_code}\n\n"
                        f"{response.text}"
                    )
                    st.metric("Latency", f"{latency_ms:.1f} ms")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Не удалось подключиться к endpoint:\n\n`{endpoint_url}`\n\n"
                    "Убедись, что ClearML Serving запущен (этап 4)."
                )
            except requests.exceptions.Timeout:
                st.error(
                    f"⏱️ Timeout ({timeout_sec}s). Попробуй увеличить лимит в настройках."
                )
            except Exception as e:
                st.error(f"Неожиданная ошибка: {e}")

# --- Статус endpoint ---
with st.expander("🔌 Проверить доступность endpoint"):
    if st.button("Ping"):
        try:
            resp = requests.get(
                endpoint_url.rstrip("/").rsplit("/serve", 1)[0] + "/",
                timeout=3,
            )
            st.success(f"Endpoint доступен (HTTP {resp.status_code})")
        except Exception as e:
            st.error(f"Endpoint недоступен: {e}")
