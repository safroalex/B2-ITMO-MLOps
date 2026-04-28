"""
ЭТАП 4: Preprocessing-скрипт для ClearML Serving

Этот файл используется clearml-serving для:
1. preprocess() — подготовка входных данных (текст → строка)
2. postprocess() — форматирование ответа (предсказание → dict)

Модель (TF-IDF + LogisticRegression в pickle) загружается автоматически.
"""

import pickle
from typing import Any

import numpy as np


class Preprocess:
    """
    Класс preprocessing для sklearn-модели в ClearML Serving.
    Метод preprocess вызывается до инференса, postprocess — после.
    """

    def preprocess(
        self,
        body: dict,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        """
        Принимает JSON-тело запроса и возвращает данные для модели.

        Ожидаемый формат запроса:
            {"text": "This movie was great!"}
        или массив:
            {"text": ["Great movie!", "Terrible film."]}
        """
        text = body.get("text", "")
        if isinstance(text, str):
            text = [text]
        return text

    def postprocess(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> dict:
        """
        Форматирует предсказание модели в JSON-ответ.

        data — массив предсказанных меток (0 или 1).
        """
        label_map = {0: "negative", 1: "positive"}

        if hasattr(data, "tolist"):
            predictions = data.tolist()
        else:
            predictions = list(data)

        if len(predictions) == 1:
            label_id = int(predictions[0])
            return {
                "label": label_map.get(label_id, str(label_id)),
                "label_id": label_id,
            }

        return {
            "predictions": [
                {"label": label_map.get(int(p), str(p)), "label_id": int(p)}
                for p in predictions
            ]
        }
