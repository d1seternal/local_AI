import pytest
import sys
from pathlib import Path
from typing import Dict, Any
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from shared.prompts import DocumentSearchPrompt
from block2_memory.main_llm_rag import  extract_keywords, clean_model_output

class TestExtractKeywords:
    
    def test_extract_keywords_basic(self):
        question = "Каковы условия аренды помещения?"
        keywords = extract_keywords(question)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(k, str) for k in keywords)
    
    def test_extract_keywords_removes_stop_words(self):
        question = "что такое машинное обучение и как оно работает"
        keywords = extract_keywords(question)
        
        stop_words = ['что', 'такое', 'и', 'как', 'оно']
        for stop in stop_words:
            assert stop not in keywords
    
    def test_extract_keywords_empty(self):
        keywords = extract_keywords("")
        assert keywords == []
    
    def test_extract_keywords_max_length(self):
        long_question = " ".join([f"слово{i}" for i in range(20)])
        keywords = extract_keywords(long_question)
        assert len(keywords) <= 5
    
    def test_extract_keywords_preserves_russian(self):
        question = "Привет мир"
        keywords = extract_keywords(question)
        assert "привет" in keywords or "мир" in keywords


class TestCleanModelOutput:
    
    def test_remove_assistant_prefix(self):
        raw = "Assistant: Это ответ ассистента"
        cleaned = clean_model_output(raw)
        assert "Assistant:" not in cleaned
        assert cleaned == "Это ответ ассистента"
    
    def test_remove_user_prefix(self):
        raw = "User: Вопрос пользователя"
        cleaned = clean_model_output(raw)
        assert "User:" not in cleaned
    
    def test_remove_special_tokens(self):
        raw = "<|system|>System<|user|>User<|assistant|>Assistant<|end|>"
        cleaned = clean_model_output(raw)
        assert "<|system|>" not in cleaned
        assert "<|user|>" not in cleaned
        assert "<|assistant|>" not in cleaned
        assert "<|end|>" not in cleaned
    
    def test_remove_multiple_artifacts(self):
        raw = "User: Вопрос\nAssistant: Ответ\nконтекст: Доп. информация"
        cleaned = clean_model_output(raw)
        assert "User:" not in cleaned
        assert "Assistant:" not in cleaned
        assert "контекст:" not in cleaned
    
    def test_preserve_normal_text(self):
        raw = "Обычный текст без артефактов"
        cleaned = clean_model_output(raw)
        assert cleaned == "Обычный текст без артефактов"
    
    def test_handle_empty_string(self):
        assert clean_model_output("") == ""
    
    def test_handle_only_artifacts(self):
        raw = "Assistant: User: контекст:"
        cleaned = clean_model_output(raw)
        assert cleaned == "" or cleaned.isspace()


class TestDocumentSearchPrompt:
    
    def test_system_prompt_not_empty(self):
        assert DocumentSearchPrompt.system_prompt
        assert len(DocumentSearchPrompt.system_prompt) > 100
    
    def test_user_prompt_format(self):
        context = "Тестовый контекст документа"
        question = "Тестовый вопрос"
        keywords = "тест, документ"
        
        formatted = DocumentSearchPrompt.user_prompt.format(
            context=context,
            question=question,
            keywords=keywords
        )
        
        assert context in formatted
        assert question in formatted
        assert keywords in formatted
    
    def test_user_prompt_contains_placeholders(self):
        assert "{context}" in DocumentSearchPrompt.user_prompt
        assert "{question}" in DocumentSearchPrompt.user_prompt
        assert "{keywords}" in DocumentSearchPrompt.user_prompt