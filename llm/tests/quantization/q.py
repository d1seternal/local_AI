import psutil
from llama_cpp import Llama

model_path = "D:\AI\Llama\models\deepseek-r1-qwen3-8b-q4_k_m.gguf"

# Замер до загрузки
mem_before = psutil.virtual_memory().used / 1024**3
print(f"До: {mem_before:.2f} GB")

# Загрузка с полным отображением в RAM
llm = Llama(
    model_path=model_path,
    use_mmap=False,      # принудительная полная загрузка
    n_ctx=4096,
    n_threads=8,
    verbose=True
)

# Замер после
mem_after = psutil.virtual_memory().used / 1024**3
print(f"После: {mem_after:.2f} GB")
print(f"Разница: {mem_after - mem_before:.2f} GB")