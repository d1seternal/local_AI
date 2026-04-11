source venv/bin/activate
# Скачать модель, если нет
MODEL_DIR="models"
MODEL_FILE="mistral-7b-instruct-v0.2.Q6_K.gguf"
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Скачиваю модель..."
    mkdir -p $MODEL_DIR
    hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF $MODEL_FILE --local-dir $MODEL_DIR
fi
python3 app.py