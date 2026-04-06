from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "agent_data"
MEMORY_PATH = PROJECT_ROOT / "agent_vector_store"
SESSIONS_DIR = PROJECT_ROOT / "sessions"
MODEL_PATH = MODEL_ROOT / "models" / "deepseek-r1-qwen3-8b-q4_k_m.gguf"
MEMORY_COLLECTION = "conversations"
DOCS_COLLECTION = "documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

MODEL_CONTEXT = 32768
MODEL_THREADS = 8
MODEL_GPU_LAYERS = 0
MODEL_TEMPERATURE = 0.5
MODEL_TOP_P = 0.9

DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_PATH.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
