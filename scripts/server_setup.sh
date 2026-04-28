#!/usr/bin/env bash
# Полная настройка сервера после docker compose up -d
# Запускать: bash scripts/server_setup.sh <ACCESS_KEY> <SECRET_KEY>
set -e

ACCESS_KEY="${1:?Usage: $0 ACCESS_KEY SECRET_KEY}"
SECRET_KEY="${2:?Usage: $0 ACCESS_KEY SECRET_KEY}"

SERVER_HOST="${SERVER_HOST:-213.165.34.52}"

echo "=== 1. Пишем clearml.conf ==="
cat > ~/clearml.conf << EOF
api {
    web_server: http://${SERVER_HOST}:8090
    api_server: http://${SERVER_HOST}:8008
    files_server: http://${SERVER_HOST}:8081
    credentials {
        "access_key" = "${ACCESS_KEY}"
        "secret_key" = "${SECRET_KEY}"
    }
}
EOF
echo "clearml.conf записан"

echo "=== 2. Устанавливаем Python зависимости ==="
python3 -m pip install --quiet --break-system-packages \
    clearml clearml-agent clearml-serving \
    scikit-learn datasets matplotlib streamlit \
    requests grpcio uvicorn fastapi

echo "=== 3. Загружаем датасет ==="
cd /root/B2-ITMO-MLOps
python3 stage1_dataset/create_dataset.py

echo "=== 4. Создаём очередь 'students' через API ==="
python3 - << 'PYEOF'
from clearml.backend_api.session.client import APIClient
client = APIClient()
try:
    res = client.queues.create(name="students")
    print(f"Очередь создана: {res.id}")
except Exception as e:
    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
        print("Очередь 'students' уже существует")
    else:
        raise
PYEOF

echo "=== 5. Запускаем ClearML Agent в фоне ==="
nohup clearml-agent daemon --queue students --detached > /tmp/clearml-agent.log 2>&1
echo "Агент запущен (логи: /tmp/clearml-agent.log)"

echo "=== 6. Ставим 2 эксперимента в очередь ==="
cd /root/B2-ITMO-MLOps/stage2_train
python3 train.py --max_features 10000 --C 1.0 --ngram_min 1 --ngram_max 1
sleep 2
python3 train.py --max_features 5000 --C 0.1 --ngram_min 1 --ngram_max 2

echo ""
echo "Эксперименты поставлены в очередь 'students'."
echo "Агент их подберёт и выполнит. Логи: /tmp/clearml-agent.log"
echo ""
echo "=== Готово! Откройте http://${SERVER_HOST}:8090 ==="
