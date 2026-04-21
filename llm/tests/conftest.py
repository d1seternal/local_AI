import pytest
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    def _create_file(filename: str, content: str = ""):
        file_path = temp_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    return _create_file


@pytest.fixture
def sample_text() -> str:
    return """
    Это тестовый документ для проверки работы парсера.
    Он содержит несколько предложений.
    И абзацы с переносами строк.
    
    Второй абзац с важной информацией.
    """


@pytest.fixture
def sample_document_chunks() -> list:
    return [
        {"text": "Чанк 1: Первое предложение.", "metadata": {"chunk_index": 0}},
        {"text": "Чанк 2: Второе предложение.", "metadata": {"chunk_index": 1}},
        {"text": "Чанк 3: Третье предложение.", "metadata": {"chunk_index": 2}},
    ]


@pytest.fixture
def sample_keywords_data() -> Dict[str, Any]:
    return {
        "question": "Каковы условия аренды и сроки платежа?",
        "expected_keywords": ["условия", "аренды", "сроки", "платежа"],
        "stop_words": ['каковы', 'и']
    }


@pytest.fixture
def sample_pdf_content() -> bytes:
    return b'%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000015 00000 n\n0000000059 00000 n\n0000000119 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n222\n%%EOF\n'

@pytest.fixture
def mock_llm():
    from unittest.mock import Mock
    mock = Mock()
    mock.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'Тестовый ответ'}}],
        'usage': {'completion_tokens': 10}
    }
    return mock


@pytest.fixture
def mock_vector_memory():
    from unittest.mock import Mock
    mock = Mock()
    mock.list_documents.return_value = []
    mock.add_document.return_value = "Добавлено 10 фрагментов"
    mock.search_with_rag.return_value = "Результаты поиска"
    mock.search_documents.return_value = []
    return mock


@pytest.fixture
def mock_config(monkeypatch):
    from shared import config
    monkeypatch.setattr(config, 'DATA_DIR', Path('/tmp/test_agent_data'))
    monkeypatch.setattr(config, 'MEMORY_PATH', Path('/tmp/test_vector_store'))
    monkeypatch.setattr(config, 'SESSIONS_DIR', Path('/tmp/test_sessions'))
    return config