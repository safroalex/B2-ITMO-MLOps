"""
ЭТАП 1: Загрузка датасета в ClearML

Скачивает IMDB sentiment dataset и создаёт версионированный ClearML Dataset.

Использование:
    pip install -r requirements.txt
    python create_dataset.py
"""

import os
import csv

from clearml import Dataset


DATASET_NAME = "imdb-sentiment"
DATASET_PROJECT = "MLOps-Course"
DATA_DIR = "./data"


def download_imdb() -> str:
    """Скачивает IMDB через HuggingFace datasets, сохраняет в CSV."""
    from datasets import load_dataset

    print("Скачиваем IMDB dataset...")
    ds = load_dataset("imdb")

    os.makedirs(DATA_DIR, exist_ok=True)

    # 5000 train + 1000 test — достаточно для курсового проекта
    train_samples = list(ds["train"])[:5000]
    test_samples = list(ds["test"])[:1000]

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    for path, samples in [(train_path, train_samples), (test_path, test_samples)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            for sample in samples:
                writer.writerow({"text": sample["text"], "label": sample["label"]})

    print(f"Train: {len(train_samples)} строк → {train_path}")
    print(f"Test:  {len(test_samples)} строк → {test_path}")
    return DATA_DIR


def create_clearml_dataset(data_dir: str) -> str:
    """Создаёт и версионирует датасет в ClearML."""
    print("\nСоздаём ClearML Dataset...")

    dataset = Dataset.create(
        dataset_name=DATASET_NAME,
        dataset_project=DATASET_PROJECT,
        dataset_version="1.0",
    )

    # Добавляем файлы по одному (надёжнее wildcard)
    dataset.add_files(path=os.path.join(data_dir, "train.csv"))
    dataset.add_files(path=os.path.join(data_dir, "test.csv"))

    # Теги для описания датасета
    dataset.add_tags(["imdb", "sentiment", "classification", "text"])

    # Загружаем файлы и фиксируем версию
    dataset.upload()
    dataset.finalize()

    print(f"Dataset ID:      {dataset.id}")
    print(f"Dataset Version: {dataset.version}")
    print("Датасет успешно загружен в ClearML!")
    print(f"\nДля обучения используй dataset_id: {dataset.id}")

    return dataset.id


if __name__ == "__main__":
    data_dir = download_imdb()
    dataset_id = create_clearml_dataset(data_dir)

