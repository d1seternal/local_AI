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


# Скачать эмбеддинговую модель, если нет
EMBEDDING_DIR="models/multilingual-e5-base"
if python -c "
from sentence_transformers import SentenceTransformer
try:
    SentenceTransformer('$EMBEDDING_DIR')
except:
    exit(1)
" 2>/dev/null | grep -q "OK"; then
else
    echo "Скачиваю модель эмбеддингов..."
    rm -rf "$EMBEDDING_DIR"
    python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/multilingual-e5-base').save('$EMBEDDING_DIR')
"
fi

python3 llm/block4_web/app.py