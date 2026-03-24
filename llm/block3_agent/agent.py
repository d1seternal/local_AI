import os
import pathlib
import shutil
import sys
import json
import re
from pathlib import Path
from datetime import datetime, date
import time
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import uuid

import subprocess

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.agents import AgentAction, AgentFinish

from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_cpp import Llama
import numpy as np

import chromadb
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


try:
    from main_llm_rag import (
        generate_with_prompts,      
        DocumentProcessor,          
        ModelBenchmark,          
        extract_keywords,            
        clean_model_output,        
        DocumentSearchPrompt,       
        LocalLLMReranker,
        load_model       
    )
    from memory import VectorMemory as RAGVectorMemory
    RAG_AVAILABLE = True
    print("RAG-модуль успешно импортирован")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"Ошибка импорта RAG: {e}")


try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("Установите: pip install python-docx")

MODEL_PATH="./models/deepseek-r1-qwen3-8b-q4_k_m.gguf"
RAG_APP_PATH = Path("main_llm_rag.py")
DATA_DIR = Path("agent_data")
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_PATH = Path("agent_vector_store")
VECTOR_DB_PATH.mkdir(exist_ok=True)


def _format_size(size: int) -> str:
    for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ТБ"

def list_files() -> str:
    try:
        files = []
        for item in DATA_DIR.iterdir():
            if item.is_file():
                size = _format_size(item.stat().st_size)
                files.append(f"{item.name} ({size})")
        return "\n".join(files) if files else "Нет файлов"
    except Exception as e:
        return f"Ошибка: {str(e)}"

class VectorMemory(RAGVectorMemory):
    def __init__(
            self, 
            persist_directory: str = "./agent_vector_store"
            ):
        super().__init__(
            persist_directory=persist_directory,
            memory_collection="conversations",
            docs_collection="documents",
            embedding_model="intfloat/multilingual-e5-base"
        )
        
        self.rag_llm = None
        self.rag_benchmark = None
    
    def _get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        if "e5" in "intfloat/multilingual-e5-base":
            prefix = "query: " if is_query else "passage: "
            text = prefix + text
        
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
   
    def set_rag_components(self, llm, benchmark):
        self.rag_llm = llm
        self.rag_benchmark = benchmark
        print("RAG компоненты установлены в VectorMemory")
    
    def add_document(self, file_path: str, doc_id: str = None) -> str:
        try:
            chunks = self.index_document(file_path, doc_id=doc_id)
            return f"Добавлено {chunks} фрагментов из {Path(file_path).name}"
        except Exception as e:
            return f"Ошибка: {str(e)}"


    def search_with_rag(self, query: str) -> str:
        try:
            clean_query = str(query).strip('"\'{}[]() ')
            
            if not clean_query:
                return "Пустой запрос"
    
            
            answer, tokens, time_taken, speed = generate_with_prompts(
                llm=self.rag_llm,        
                query_text=clean_query,
                benchmark=self.rag_benchmark,
                memory=self,               
                session_id="agent_search",
                use_docs=True,
                reranker=None
            )
            
            if not answer:
                return f"По запросу '{clean_query}' ничего не найдено"
            
            if tokens > 0:
                return f"{answer}\n\n_[{tokens} токенов | {time_taken:.1f} сек]_"
            
            return answer
            
        except ImportError as e:
            print(f"Ошибка импорта RAG: {e}")
            return self.search(query)
        except Exception as e:
            print(f"Ошибка RAG поиска: {e}")
            return f"Ошибка: {str(e)}"  
        

    def search(self, query: str, k: int = 3, score_threshold: float = 0.5) -> str:
        """
        Поиск в векторной памяти
        
        Args:
            query: str - поисковый запрос
            k: int - количество результатов
            score_threshold: float - порог релевантности
        
        Returns:
            str - отформатированные результаты
        """
        try:
            clean_query = str(query).strip('"\'{}[]() ')
            
            if not clean_query:
                return "Пустой запрос"
            
            query_embedding = self._get_embedding(clean_query, is_query=True)
            results = self.docs_collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 2  
            )
            
            if not results['ids'][0]:
                return f"По запросу '{clean_query}' ничего не найдено"
            
            formatted = []
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                score = max(0, 1 - distance / 2)
                
                if score >= score_threshold:
                    metadata = results['metadatas'][0][i]
                    formatted.append({
                        "text": results['documents'][0][i],
                        "metadata": metadata,
                        "relevance_score": score
                    })

            formatted.sort(key=lambda x: x['relevance_score'], reverse=True)
            formatted = formatted[:k]

            output = [f"Результаты поиска: '{clean_query}'\n"]
            
            for i, r in enumerate(formatted, 1):
                relevance = r['relevance_score'] * 100
                filename = r['metadata'].get('filename', 'неизвестно')
                chunk_idx = r['metadata'].get('chunk_index', 0)
                total = r['metadata'].get('chunk_total', 1)
                
                text = r['text'][:700] + ("..." if len(r['text']) > 700 else "")
                
                output.append(f"{i}. **[{relevance:.0f}%]** {filename} (фрагмент {chunk_idx+1}/{total})")
                output.append(f"  {text}\n")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Ошибка при поиске: {str(e)}"
    
    def list_documents(self) -> str:
        try:
            results = self.docs_collection.get()
            
            docs = {}
            for metadata in results['metadatas']:
                doc_id = metadata.get('doc_id')
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "filename": metadata.get('filename', 'unknown'),
                        "chunks": 0,
                        "timestamp": metadata.get('timestamp', 0)
                    }
                if doc_id:
                    docs[doc_id]['chunks'] += 1
            
            if not docs:
                return "Векторная память пуста"
            
            output = ["Документы в памяти: "]
            for doc_id, info in sorted(docs.items(), key=lambda x: x[1]['timestamp'], reverse=True):
                timestamp = datetime.fromtimestamp(info['timestamp']).strftime('%Y-%m-%d %H:%M')
                output.append(f"  • {info['filename']} ({info['chunks']} фрагментов) - {timestamp}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Ошибка: {str(e)}"
        
vector_memory = VectorMemory(persist_directory="./agent_vector_store")

class AgentLLM(LLM):
    shared_llm: Any = None

    def __init__(self, shared_llm):
        super().__init__()
        self.shared_llm = shared_llm
    
    @property
    def _llm_type(self) -> str:
        return "agent_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = self.shared_llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.5,
                stop=stop or ["Observation:", "User:", "Assistant:"]
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Ошибка: {str(e)}"

@tool
def vector_add(filepath: str) -> str:
    """
    Добавить файл в векторную память для последующего поиска.
    
    Аргументы:
        filepath: str - имя файла (например, "document.txt" или "отчет.docx")
    
    Возвращает:
        str - результат операции с количеством добавленных фрагментов
    """
    return vector_memory.add_document(filepath)

# @tool
# def vector_search(query: str) -> str:
#     """
#     Поиск информации в векторной памяти.
    
#     Аргументы:
#         query: str - поисковый запрос
    
#     Возвращает:
#         str - найденные фрагменты с релевантностью
#     """
#     return vector_memory.search(query, k=3)

@tool
def vector_list() -> str:
    """
    Показать список всех документов в векторной памяти.
    
    Возвращает:
        str - список документов с количеством фрагментов
    """
    return vector_memory.list_documents()


@tool
def search_documents(query: str) -> str:
    """
     Поиск информации в базе знаний (RAG-приложение).
     Использует все проиндексированные документы для поиска информации.
    
     Аргументы:
         query: str - поисковый запрос
    
     Возвращает:
         str - найденные фрагменты с указанием источника
     """

    return vector_memory.search_with_rag(query)


@tool
def write_file(filepath: str, content: str, append: bool = False) -> str:
    """
    Записать или добавить текст в файл. Если файл уже существует в памяти,
    то автоматическая переиндексация.
    
    Аргументы:
        filepath: str - путь к файлу
        content: str - текст для записи
        append: bool - True - добавить в конец, False - перезаписать
    
    Возвращает:
        str - результат операции
    """
    try:
        filename = Path(str(filepath).strip('"\'{}[]() ')).name
        full_path = DATA_DIR / filename
        
        mode = 'a' if append else 'w'
        action = "Добавлено в" if append else "Записано в"
        
        existed_before = full_path.exists()
        with open(full_path, mode, encoding='utf-8') as f:
            f.write(content + ("\n" if not content.endswith("\n") else ""))
        
        
        stats = full_path.stat()
        result_msg = f"{action} {filename} ({len(content)} символов)\n Размер: {_format_size(stats.st_size)}"
        
        if existed_before:
            print(f"Файл {filename} изменен, обновляем индекс...")
            try:
                docs = vector_memory.list_documents()
                doc_to_delete = None
                for doc in docs:
                    if doc.get('filename') == filename:
                        doc_to_delete = doc.get('doc_id')
                        break
                
                if doc_to_delete:
                    vector_memory.delete_document(doc_to_delete)
                    result_msg += f"\nСтарая версия удалена из памяти"
                
                chunks = vector_memory.add_document(str(full_path))
                result_msg += f"\nОбновленный файл добавлен в память ({chunks} фрагментов)"
                
            except Exception as e:
                result_msg += f"\nНе удалось обновить индекс: {e}"
        
        return result_msg
        
    except Exception as e:
        return f"Ошибка записи: {str(e)}"


@tool
def execute_python(code: str) -> str:
    """
    Выполнить Python код.
    
    Аргументы:
        code: str - Python код для выполнения
    
    Возвращает:
        str - результат выполнения
    """
    try:
        clean_code = str(code).strip('"\'')
        
        temp_file = DATA_DIR / f"_temp_{uuid.uuid4().hex[:8]}.py"
        
        full_code = f"""# -*- coding: utf-8 -*-
            import sys
            import json
            import os
            import math
            import random
            import traceback
            from datetime import datetime
            from pathlib import Path

            sys.path.insert(0, r'{DATA_DIR.absolute()}')

            try:
                {chr(10).join(['    ' + line for line in clean_code.split(chr(10))])}
            except Exception as e:
                print(f"Ошибка: {{e}}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        """
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        result = subprocess.run(
            [sys.executable, str(temp_file)],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
            cwd=str(DATA_DIR)
        )
        
        temp_file.unlink(missing_ok=True)
        
        output = []
        if result.stdout:
            output.append(f"Результат:\n{result.stdout}")
        if result.stderr:
            output.append(f"Ошибки:\n{result.stderr}")
        
        if not output:
            return "Код выполнен без вывода"
        
        return "\n".join(output)
        
    except subprocess.TimeoutExpired:
        return "Превышено время выполнения"
    except Exception as e:
        return f"Ошибка выполнения: {str(e)}"


def create_agent():
    print("Загрузка GGUF модели...")
    shared_llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        n_gpu_layers=0,
        temperature=0.5,
        top_p=0.9,
        verbose=False
    )
    print("Модель загружена")
    
    class Benchmark:
        pass
    benchmark = Benchmark()
    vector_memory.set_rag_components(shared_llm, benchmark)
    llm = AgentLLM(shared_llm=shared_llm)

    tools = [
        vector_add,        
        vector_list,          
        search_documents,    
        write_file,          
        execute_python        
    ]

    prompt = ChatPromptTemplate.from_template(
        """Ты - ползеный AI-ассистент для работы с документами и файлами. Сегодня {date}.

Доступные инструменты:
{tools}
Имена инструментов: {tool_names}
Используй следующий формат:
Question: {input}
Thought: (твои мысли и рассуждения на тему того, что нужно сделать)
Action: (название инструмента из списка)
Action Input: (входные данные для инструмента в формате JSON)
Observation: (результат выполнения инструмента)
(ты можешь повторять Thought/Action/Action Input/Observation несколько раз)
Thought: (я понял результат, теперь могу ответить)
Final Answer: (ответ пользователю)
Если ты нашел ответ и готов ответить пользователю:
FINAL ANSWER: [твой ответ] (например, "файл добавлен в память" или "Результаты поиска: <фрагменты>"), то завершай свою генерацию и ожидай новый запрос пользователя. Не нужно размышлять дальше, доходя до лимита итераций.

Начало работы:
Question: {input}
Thought: {agent_scratchpad}"""
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )
    
    return agent_executor


def main():
    if not Path(MODEL_PATH).exists():
        print(f"Модель не найдена: {MODEL_PATH}")
        print("Укажите правильный путь к модели")
        return
    
    print(f"Загрузка агента...")
    agent = create_agent()
    print("Агент готов!")
    
    print("\n")
    print("Интерактивный режим. Команды:")
    print("  /help - показать инструменты")
    print("  /files - список файлов")
    print("  /exit - выход")
    print("\n")
    
    while True:
        try:
            query = input("\n > ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['/exit', '/quit']:
                break
            
            if query.lower() == '/help':
                print("  • vector_list - список файлов")
                print("  • vector_add - добавление файла в память")
                print("  • search_documents - искать информацию через RAG")
                print("  • write_file - записать данные в файл")
                print("  • execute_python - выполнить Python-код")
                continue
            
            if query.lower() == '/files':
                print(list_files())
                continue
  
            print("\nОбработка запроса...")
            response = agent.invoke({"input": query, 
                                    "date": date.today().strftime('%d.%m.%Y')})
            print(f"\n{response['output']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
