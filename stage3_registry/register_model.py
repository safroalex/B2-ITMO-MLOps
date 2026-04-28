"""
ЭТАП 3: Регистрация лучшей модели в ClearML Model Registry

Находит завершённые эксперименты (train.py), выбирает лучший по F1,
публикует OutputModel в Registry с тегами и метаданными.

Требования:
  - Должны быть завершены минимум 2 эксперимента (train.py)
  - Каждый эксперимент должен был создать OutputModel (train.py это делает)

Использование:
    python register_model.py
"""

from clearml import Task
from clearml.model import Model


TASK_PROJECT = "MLOps-Course"
TASK_NAME = "imdb-train"
REGISTRY_MODEL_NAME = "imdb-sentiment-model"


def find_best_task():
    """Находит завершённый эксперимент с наилучшим F1."""
    print(f"Ищем завершённые задачи '{TASK_NAME}' в проекте '{TASK_PROJECT}'...")
    tasks = Task.get_tasks(
        project_name=TASK_PROJECT,
        task_name=TASK_NAME,
        task_filter={"status": ["completed"]},
    )

    if not tasks:
        raise RuntimeError(
            f"Не найдено завершённых задач '{TASK_NAME}'.\n"
            "Сначала запусти stage2_train/train.py дважды и дождись "
            "выполнения агентом."
        )

    print(f"Найдено задач: {len(tasks)}")

    best_task = None
    best_f1 = -1.0

    for t in tasks:
        scalars = t.get_last_scalar_metrics()
        try:
            f1 = scalars["metrics"]["f1"]["last"]
        except (KeyError, TypeError):
            print(f"  Task {t.id}: метрика f1 не найдена, пропускаем")
            continue

        accuracy = scalars.get("metrics", {}).get("accuracy", {}).get("last", None)
        print(f"  Task {t.id}: f1={f1:.4f}"
              + (f", accuracy={accuracy:.4f}" if accuracy else ""))

        if f1 > best_f1:
            best_f1 = f1
            best_task = t

    if best_task is None:
        raise RuntimeError(
            "Ни одна задача не содержит метрику 'f1'.\n"
            "Убедись что train.py корректно логирует метрики."
        )

    print(f"\nЛучшая задача: {best_task.id}  (f1={best_f1:.4f})")
    return best_task, best_f1


def register_model(best_task: Task, best_f1: float) -> str:
    """
    Публикует OutputModel из лучшего эксперимента в Model Registry.

    train.py создаёт OutputModel через OutputModel(task=task, ...).
    Здесь мы его находим через task.get_models()["output"] и вызываем .publish().
    """
    # Получаем список OutputModel из задачи
    task_models = best_task.get_models()
    output_models = task_models.get("output", [])

    if not output_models:
        raise RuntimeError(
            f"В задаче {best_task.id} нет OutputModel.\n"
            "Убедись что в train.py используется OutputModel(task=task, ...)."
        )

    # Берём первую модель (одна на задачу)
    model_ref = output_models[0]
    model_id = model_ref.id
    print(f"Найден OutputModel: {model_id}")

    # Загружаем полный объект модели
    model = Model(model_id=model_id)

    # Получаем гиперпараметры для тегов
    raw_params = best_task.get_parameters_as_dict()
    hyperparams = raw_params.get("hyperparameters", {})

    # Получаем accuracy из метрик
    scalars = best_task.get_last_scalar_metrics()
    accuracy = scalars.get("metrics", {}).get("accuracy", {}).get("last", None)

    # Теги — будут видны в Registry
    tags = [
        f"f1={best_f1:.4f}",
        f"C={hyperparams.get('C', 'unknown')}",
        f"max_features={hyperparams.get('max_features', 'unknown')}",
        "sentiment",
        "sklearn",
        "production",
    ]
    if accuracy is not None:
        tags.append(f"accuracy={accuracy:.4f}")

    model.add_tags(tags)

    # Переименовываем модель
    model.name = REGISTRY_MODEL_NAME

    # publish() → модель попадает в раздел Models / Registry со статусом Published
    model.publish()

    print(f"\nМодель зарегистрирована в Registry:")
    print(f"  Model ID:  {model_id}")
    print(f"  Имя:       {REGISTRY_MODEL_NAME}")
    print(f"  F1:        {best_f1:.4f}")
    if accuracy:
        print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Теги:      {tags}")
    print(f"\nПроверь: Web UI → Models → статус 'Published'")

    return model_id


if __name__ == "__main__":
    best_task, best_f1 = find_best_task()
    model_id = register_model(best_task, best_f1)
    print(f"\nModel ID для serving: {model_id}")
