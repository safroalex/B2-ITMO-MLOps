"""
ЭТАП 2: Обучение модели через ClearML Agent

Запускается локально → ставится в очередь "students" → выполняется агентом.

Использование:
    # Эксперимент 1 (дефолтные параметры)
    python train.py

    # Эксперимент 2 (другие гиперпараметры)
    python train.py --max_features 5000 --C 0.1 --ngram_max 2

    # Запустить локально без агента (для отладки)
    python train.py --local
"""

import argparse
import csv
import os
import pickle
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from clearml import Dataset, OutputModel, Task


DATASET_PROJECT = "MLOps-Course"
DATASET_NAME = "imdb-sentiment"
TASK_PROJECT = "MLOps-Course"
TASK_NAME = "imdb-train"
QUEUE_NAME = "students"


def parse_args():
    parser = argparse.ArgumentParser(description="Train IMDB sentiment classifier")
    parser.add_argument("--max_features", type=int, default=10000,
                        help="TF-IDF max features")
    parser.add_argument("--ngram_min", type=int, default=1,
                        help="TF-IDF ngram range min")
    parser.add_argument("--ngram_max", type=int, default=1,
                        help="TF-IDF ngram range max")
    parser.add_argument("--C", type=float, default=1.0,
                        help="LogisticRegression regularization parameter C")
    parser.add_argument("--max_iter", type=int, default=300,
                        help="LogisticRegression max iterations")
    parser.add_argument("--local", action="store_true",
                        help="Запустить локально без отправки в очередь агента")
    return parser.parse_args()


def load_csv(path: str):
    """Загружает CSV-датасет в списки текстов и меток."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def get_current_git_commit() -> str:
    """Возвращает текущий git commit hash или пустую строку."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def log_confusion_matrix(task: Task, y_true, y_pred):
    """Строит confusion matrix и логирует как изображение и как таблицу в ClearML."""
    labels = ["negative", "positive"]
    cm = confusion_matrix(y_true, y_pred)

    # Логируем как встроенную таблицу ClearML
    task.get_logger().report_confusion_matrix(
        title="Confusion Matrix",
        series="test",
        matrix=cm,
        xaxis="Predicted",
        yaxis="Actual",
        xlabels=labels,
        ylabels=labels,
        iteration=0,
    )

    # Логируем также как изображение (требование задания)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Test set)")
    plt.tight_layout()

    tmp_path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
    fig.savefig(tmp_path, dpi=100)
    task.get_logger().report_image(
        title="Confusion Matrix",
        series="test",
        local_path=tmp_path,
        iteration=0,
    )
    plt.close(fig)
    os.unlink(tmp_path)


def main():
    args = parse_args()

    # Создаём ClearML Task
    task = Task.init(
        project_name=TASK_PROJECT,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )

    # Логируем гиперпараметры (будут видны в UI и доступны для клонирования)
    hyperparams = {
        "max_features": args.max_features,
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "C": args.C,
        "max_iter": args.max_iter,
        "model": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "solver": "lbfgs",
    }
    task.connect(hyperparams, name="hyperparameters")

    # Фиксируем git commit
    git_commit = get_current_git_commit()
    if git_commit:
        task.set_parameter("git/commit", git_commit)
        print(f"Git commit: {git_commit}")

    # Если не --local, ставим задачу в очередь для агента и выходим локально.
    # Агент подберёт задачу, переклонирует окружение и выполнит код ниже.
    if not args.local:
        print(f"Ставим задачу в очередь: {QUEUE_NAME}")
        task.execute_remotely(queue_name=QUEUE_NAME, clone=False, exit_process=True)

    # =========================================================
    # Всё ниже выполняется ТОЛЬКО на агенте (или при --local)
    # =========================================================

    logger = task.get_logger()

    # Загружаем датасет из ClearML (по имени и проекту)
    print("Загружаем датасет из ClearML...")
    dataset = Dataset.get(
        dataset_project=DATASET_PROJECT,
        dataset_name=DATASET_NAME,
        only_completed=True,
        only_published=False,
    )
    dataset_path = dataset.get_local_copy()
    print(f"Датасет загружен в: {dataset_path}")
    # Логируем dataset_id для воспроизводимости
    task.set_parameter("dataset/id", dataset.id)
    task.set_parameter("dataset/version", str(dataset.version))

    train_path = os.path.join(dataset_path, "train.csv")
    test_path = os.path.join(dataset_path, "test.csv")

    X_train, y_train = load_csv(train_path)
    X_test, y_test = load_csv(test_path)
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # ---- Строим Pipeline (TF-IDF + LogisticRegression) ----
    # Pipeline сохраняется единым объектом и корректно обрабатывается ClearML Serving
    print("Обучаем модель...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=hyperparams["max_features"],
            ngram_range=(hyperparams["ngram_min"], hyperparams["ngram_max"]),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=hyperparams["C"],
            max_iter=hyperparams["max_iter"],
            solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)

    # ---- Метрики ----
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1:       {f1:.4f}")

    logger.report_scalar("metrics", "accuracy", iteration=0, value=accuracy)
    logger.report_scalar("metrics", "f1", iteration=0, value=f1)

    # ---- Confusion matrix (изображение + таблица) ----
    log_confusion_matrix(task, y_test, y_pred)

    # ---- Сохраняем pipeline в pickle ----
    model_path = os.path.join(tempfile.gettempdir(), "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    # ---- OutputModel → попадает в Models tab → нужен для Registry ----
    # Это ОБЯЗАТЕЛЬНО для этапа 3 (Model Registry)
    output_model = OutputModel(
        task=task,
        framework="scikit-learn",
        name="imdb-sentiment",
        label_enumeration={"negative": 0, "positive": 1},
    )
    output_model.update_weights(
        weights_filename=model_path,
        target_filename="model.pkl",
        auto_delete_file=False,
        iteration=0,
    )
    output_model.add_tags([
        f"f1={f1:.4f}",
        f"accuracy={accuracy:.4f}",
        f"C={hyperparams['C']}",
        f"max_features={hyperparams['max_features']}",
        "sklearn",
        "sentiment",
    ])

    # ---- Artifact (дополнительно, для воспроизводимости) ----
    task.upload_artifact(
        name="model",
        artifact_object=model_path,
        metadata={
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "output_model_id": output_model.id,
        },
    )

    # Удаляем временный файл
    os.unlink(model_path)

    print(f"\nМодель сохранена:")
    print(f"  OutputModel ID: {output_model.id}")
    print(f"  Artifact:       'model'")
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  F1:             {f1:.4f}")
    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()

