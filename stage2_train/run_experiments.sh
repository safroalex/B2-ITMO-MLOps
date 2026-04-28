#!/usr/bin/env bash
#
# Запуск 2 экспериментов с разными гиперпараметрами для демонстрации
# различий в ClearML UI (требование этапа 2: минимум 2 эксперимента)
#
# Использование:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Задачи ставятся в очередь "students" и выполняются агентом.
# Прогресс можно отслеживать в Web UI → Experiments.

set -e

echo "============================================"
echo "ЭКСПЕРИМЕНТ 1: Базовая конфигурация"
echo "  max_features=10000, C=1.0, ngram=(1,1)"
echo "============================================"
python train.py \
    --max_features 10000 \
    --C 1.0 \
    --ngram_min 1 \
    --ngram_max 1

echo ""
echo "Задача 1 поставлена в очередь 'students'."
echo ""

# Небольшая пауза чтобы задачи получили разные timestamp
sleep 2

echo "============================================"
echo "ЭКСПЕРИМЕНТ 2: Альтернативная конфигурация"
echo "  max_features=5000, C=0.1, ngram=(1,2)"
echo "============================================"
python train.py \
    --max_features 5000 \
    --C 0.1 \
    --ngram_min 1 \
    --ngram_max 2

echo ""
echo "Задача 2 поставлена в очередь 'students'."
echo ""
echo "Обе задачи ждут выполнения агентом."
echo "Открой Web UI → Experiments для отслеживания прогресса."
echo "После завершения запусти: python ../stage3_registry/register_model.py"
