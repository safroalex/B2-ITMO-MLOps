# B2-ITMO-MLOps — Курсовой проект

Сервис классификации текстов (IMDB sentiment analysis) через полный ML lifecycle на ClearML.

**Стек:** Python, sklearn (TF-IDF + LogisticRegression), ClearML, Streamlit

---

## Структура проекта

```
├── docker-compose.yml          # Вся инфраструктура (Server + Serving)
├── requirements.txt            # Все Python-зависимости
├── .env.example                # Шаблон переменных окружения
├── stage1_dataset/
│   └── create_dataset.py       # ЭТАП 1: загрузка IMDB в ClearML Dataset
├── stage2_train/
│   ├── train.py                # ЭТАП 2: обучение через ClearML Agent
│   └── run_experiments.sh      # Запуск 2 экспериментов одной командой
├── stage3_registry/
│   └── register_model.py       # ЭТАП 3: публикация модели в Model Registry
├── stage4_serving/
│   └── preprocess.py           # ЭТАП 4: preprocess/postprocess для Serving
└── stage5_ui/
    └── app.py                  # ЭТАП 5: Streamlit UI
```

---

## Быстрый старт

```bash
# 1. Клонировать и установить зависимости
git clone <repo>
cd B2-ITMO-MLOps
pip install -r requirements.txt

# 2. Поднять ClearML Server
docker compose up -d

# 3. Получить credentials и настроить SDK
#    → http://localhost:8090 → Settings → Workspace → Create new credentials
clearml-init

# 4. Создать очередь и запустить агент (в отдельном терминале)
#    → http://localhost:8090 → Settings → Queues → New Queue → "students"
clearml-agent daemon --queue students

# 5. Загрузить датасет
python stage1_dataset/create_dataset.py

# 6. Запустить 2 эксперимента
cd stage2_train && chmod +x run_experiments.sh && ./run_experiments.sh && cd ..

# 7. Зарегистрировать лучшую модель
python stage3_registry/register_model.py

# 8. Задеплоить serving (нужны credentials в .env)
cp .env.example .env  # заполни CLEARML_API_ACCESS_KEY и CLEARML_API_SECRET_KEY
docker compose --profile serving up -d
# Затем:
clearml-serving create --name "imdb-serving"   # сохрани SERVICE_ID
clearml-serving model add \
    --engine sklearn \
    --endpoint "sentiment" \
    --preprocess "$(pwd)/stage4_serving/preprocess.py" \
    --model-id <MODEL_ID> \
    --service-id <SERVICE_ID>

# 9. Запустить UI
streamlit run stage5_ui/app.py
```

---

## Порты

| Сервис | URL |
|--------|-----|
| ClearML Web UI | http://localhost:8090 |
| ClearML API | http://localhost:8008 |
| ClearML Files | http://localhost:8081 |
| Serving (inference) | http://localhost:8082 |
| Streamlit UI | http://localhost:8501 |

---

## ЭТАП 0. Подготовка инфраструктуры

### 1. Запустить ClearML Server

```bash
docker compose up -d
docker compose ps   # убедиться что все сервисы running
```

### 2. Получить credentials

1. Открыть http://localhost:8090
2. Зарегистрироваться (первый пользователь — admin)
3. **Settings → Workspace → Create new credentials**
4. Скопировать `access_key` и `secret_key`

### 3. Настроить SDK

```bash
clearml-init
# API server URL:   http://localhost:8008
# Web server URL:   http://localhost:8090
# Files server URL: http://localhost:8081
# Access key / Secret key: <из UI>
```

### 4. Запустить ClearML Agent

```bash
# Создать очередь "students":
# Web UI → Settings → Queues → New Queue → "students"

# Запустить агент (в отдельном терминале, держать запущенным)
clearml-agent daemon --queue students
```

**Проверка:** Web UI → **Orchestration → Workers** → виден агент.

---

## ЭТАП 1. Загрузка датасета

```bash
python stage1_dataset/create_dataset.py
```

Скачивает IMDB (5000 train + 1000 test), загружает в ClearML Dataset.  
**Проверка:** Web UI → **Datasets** → `imdb-sentiment` с версией `1.0`

---

## ЭТАП 2. Обучение через Agent

```bash
cd stage2_train
./run_experiments.sh   # запускает 2 эксперимента с разными параметрами
cd ..
```

Или по одному:
```bash
python stage2_train/train.py                                            # exp 1
python stage2_train/train.py --max_features 5000 --C 0.1 --ngram_max 2 # exp 2
```

**Проверка:** Web UI → **Experiments** → 2 задачи, выполненные агентом, с разными params/metrics/confusion matrix.

---

## ЭТАП 3. Регистрация модели

```bash
python stage3_registry/register_model.py
```

Находит лучший эксперимент по F1, публикует модель в Registry.  
**Проверка:** Web UI → **Models** → `imdb-sentiment-model` со статусом `Published`.

---

## ЭТАП 4. Inference Endpoint

ClearML Serving загружает модель из Registry — локальный `.pkl` не используется.

### Шаг 1. Создать Serving Service

```bash
clearml-serving create --name "imdb-serving"
# Вывод: "New serving service created: <SERVICE_ID>"
# Сохрани SERVICE_ID
```

### Шаг 2. Задеплоить модель

```bash
# MODEL_ID — из вывода register_model.py или Web UI → Models → imdb-sentiment-model
clearml-serving model add \
    --engine sklearn \
    --endpoint "sentiment" \
    --preprocess "$(pwd)/stage4_serving/preprocess.py" \
    --model-id <MODEL_ID> \
    --service-id <SERVICE_ID>
```

### Шаг 3. Запустить сервер

```bash
# Вариант А: локально через uvicorn (рекомендуется для разработки)
pip install grpcio uvicorn fastapi
CLEARML_SERVING_TASK_ID=<SERVICE_ID> CLEARML_BKG_THREAD_REPORT=1 \
  python -m uvicorn clearml_serving.serving.main:app --host 0.0.0.0 --port 8082

# Вариант Б: через Docker Compose (после заполнения .env)
docker compose --profile serving up -d
```

Сервер на **http://localhost:8082**

### Проверка (примеры запросов с обоими классами)

```bash
# Positive
curl -X POST http://localhost:8082/serve/sentiment \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic! Best film I have seen in years."}'
# → {"label": "positive", "label_id": 1}

# Negative
curl -X POST http://localhost:8082/serve/sentiment \
     -H "Content-Type: application/json" \
     -d '{"text": "Terrible movie. Complete waste of time. Would not recommend."}'
# → {"label": "negative", "label_id": 0}

# Batch
curl -X POST http://localhost:8082/serve/sentiment \
     -H "Content-Type: application/json" \
     -d '{"text": ["Amazing film!", "Horrible, avoid this movie."]}'
# → {"predictions": [{"label": "positive", "label_id": 1}, {"label": "negative", "label_id": 0}]}
```

### Проверка статуса

```bash
clearml-serving status --service-id <SERVICE_ID>
```

### Troubleshooting

| Проблема | Решение |
|----------|---------|
| `Address already in use` | `CLEARML_SERVING_PORT=8083 clearml-serving serve ...` |
| `Model not found` | Убедись что Model ID верный и модель опубликована |
| `cannot call predict on dict` | train.py сохраняет Pipeline — проверь что не изменено |
| `preprocess import error` | Укажи абсолютный путь к `preprocess.py` |

---

## ЭТАП 5. UI

```bash
streamlit run stage5_ui/app.py
# → http://localhost:8501
```

UI работает через HTTP к serving (http://localhost:8082), модель не загружается напрямую.
