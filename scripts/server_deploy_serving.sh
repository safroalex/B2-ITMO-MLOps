#!/usr/bin/env bash
# Финальный деплой: serving + streamlit UI
# Запускать после server_setup.sh и регистрации модели
# Использование: bash scripts/server_deploy_serving.sh <SERVICE_ID> <MODEL_ID>
set -e

SERVICE_ID="${1:?Usage: $0 SERVICE_ID MODEL_ID}"
MODEL_ID="${2:?Usage: $0 SERVICE_ID MODEL_ID}"
SERVER_HOST="${SERVER_HOST:-213.165.34.52}"

echo "=== 1. Добавляем модель в serving ==="
clearml-serving --id "$SERVICE_ID" model add \
    --engine sklearn \
    --endpoint "sentiment" \
    --preprocess "/root/B2-ITMO-MLOps/stage4_serving/preprocess.py" \
    --model-id "$MODEL_ID"

echo "=== 2. Запускаем serving сервер ==="
nohup env \
    CLEARML_SERVING_TASK_ID="$SERVICE_ID" \
    CLEARML_BKG_THREAD_REPORT=1 \
    python3 -m uvicorn clearml_serving.serving.main:app \
    --host 0.0.0.0 --port 8082 \
    > /tmp/clearml-serving.log 2>&1 &
echo "Serving запущен (логи: /tmp/clearml-serving.log)"
sleep 5

echo "=== 3. Проверяем serving ==="
curl -sf -X POST http://localhost:8082/serve/sentiment \
    -H "Content-Type: application/json" \
    -d '{"text": "fantastic movie!"}' && echo ""

echo "=== 4. Запускаем Streamlit UI ==="
nohup env SERVING_ENDPOINT="http://${SERVER_HOST}:8082/serve/sentiment" \
    python3 -m streamlit run /root/B2-ITMO-MLOps/stage5_ui/app.py \
    --server.port 8501 --server.address 0.0.0.0 \
    > /tmp/streamlit.log 2>&1 &
echo "Streamlit запущен (логи: /tmp/streamlit.log)"
sleep 3

echo ""
echo "=== Все сервисы запущены ==="
echo "ClearML UI:  http://${SERVER_HOST}:8080"
echo "Serving:     http://${SERVER_HOST}:8082/serve/sentiment"
echo "Streamlit:   http://${SERVER_HOST}:8501"
