#!/bin/bash
set -e

source venv/bin/activate
# Скачать модель, если нет
MODEL_DIR="models"
MODEL_FILE="deepseek-r1-qwen3-8b-q4_k_m.gguf"
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Скачиваю модель..."
    mkdir -p $MODEL_DIR
    hf download muranAI/DeepSeek-R1-0528-Qwen3-8B-GGUF $MODEL_FILE --local-dir $MODEL_DIR
fi
python3 llm/block4_web/app.py