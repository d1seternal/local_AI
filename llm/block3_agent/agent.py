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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.__init__ import (
    VectorMemory as BaseVectorMemory,
    SessionManager,
    DocumentProcessor,
    MODEL_PATH,
    DATA_DIR,
    EMBEDDING_MODEL,
    MODEL_TOP_P,
    SEED,
    MODEL_CONTEXT,
    MODEL_THREADS,
    MODEL_TEMPERATURE,
    MODEL_GPU_LAYERS
)
from block2_memory.main_llm_rag import (
        generate_with_prompts,               
        ModelBenchmark,          
        extract_keywords,            
        clean_model_output,            
        load_model       
    )

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("Установите: pip install python-docx")

agent_executor = None
current_session_id = None

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

class VectorMemory(BaseVectorMemory):
    def set_rag_components(self, llm, benchmark):
        self.rag_llm = llm
        self.rag_benchmark = benchmark
        print("RAG компоненты установлены.")

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
        
vector_memory = VectorMemory()

session_memory = SessionManager()

class AgentLLM(LLM):
    shared_llm: Any = None

    def __init__(self, shared_llm):
        super().__init__()
        self.shared_llm = shared_llm    
    @property
    def _llm_type(self) -> str:
        return "agent_llm"
    
    def _clean_output(self, text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
        if not text:
            return ""
        
        return text
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = self.shared_llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=MODEL_TEMPERATURE,
                stop=stop or ["User:", "Final Answer:", "Assistant:"]
            )
           
            output = response['choices'][0]['message']['content'].strip()
            output = self._clean_output(output)
            return output
        
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

@tool
def vector_list() -> str:
    """
    Показать список всех документов в векторной памяти.
    
    Возвращает:
        str - список документов с количеством фрагментов
    """
    docs = vector_memory.list_documents()
    if not docs:
        return "Нет документов"
    return "\n".join([f"• {d['filename']} ({d['chunks']} фрагментов)" for d in docs])


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
    """Выполнить Python код"""
    try:
        clean_code = code.strip('"\'')
        
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

def main():
{chr(10).join(['    ' + line for line in clean_code.split(chr(10))])}

if __name__ == "__main__":
    main()
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
    global agent_executor
    print("Загрузка GGUF модели...")
    shared_llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=MODEL_CONTEXT,
        n_threads=MODEL_THREADS,
        n_gpu_layers=MODEL_GPU_LAYERS,
        temperature=MODEL_TEMPERATURE,
        top_p=MODEL_TOP_P,
        verbose=False
    )
    print("Модель загружена")
    
    class B_Benchmark:
        def add_query_result(self, *args, **kwargs): pass

    benchmark = B_Benchmark()
    vector_memory.set_rag_components(shared_llm, benchmark)
    llm = AgentLLM(shared_llm=shared_llm)

    tools = [
        vector_add,        
        vector_list,          
        search_documents,    
        write_file,          
        execute_python       
    ]

    for tool in tools:
        tool.return_direct = False

    prompt = ChatPromptTemplate.from_template(
        """Ты - полезный AI-ассистент для работы с документами и файлами. Сегодня {date}.

Никогда не пиши комментарии, заметки или пояснения, выходящие за рамки этого формата.
Никогда не пиши ничего вроде "Подожду результат..." и другие комментарии после Action Input
После Action Input идет исключительно Observation с полученным результатом
Если какой-нибудь инструмент был выполнен успешно, вернув результат, то не делай его снова. Просто бери готовый результат
Если пользователь интересуется информацией не по документам, или в запросе отсутствует название файла, то отвечай в свободном виде, исходя из своих имеющихся данных, не ищи информацию в документах.
Если информация для ответа не была найдена, то выводи одно сообщение "информация не найдена". Не нужно выводить шаблоны для ответа из промптов.
Не используй теги <think> или </think>
Не добавляй рассуждения в тегах
Не используй **жирный текст** или маркдаун
Не добавляй лишние слова перед Action
На запрос пользователя обязательно нужно вернуть четкий ответ. Не добавляй других слов и комментариев. Если идет последний цикл генерации, а конкретный ответ не найден, то выводи информацию "данные не найдены"
Если ты нашел ответ и готов ответить пользователю:
- Final Answer: [твой ответ] (например, "файл добавлен в память" или "Результаты поиска: <фрагменты>"), то завершай свою генерацию и ожидай новый запрос пользователя. Не нужно размышлять дальше, доходя до лимита итераций.
Final Answer должен быть кратким
Не выводи JSON в Final Answer
Не рассуждай в Final Answer

Доступные инструменты:
{tools}
Имена инструментов: {tool_names}
Генерировать верную информацию тебе помогут следующие принципы:
Question: {input}
Thought: (твои мысли и рассуждения на тему того, что нужно сделать)
Action: (название инструмента из списка)
Action Input: (входные данные для инструмента в формате JSON)
Observation: (результат выполнения инструмента)
(ты можешь повторять Thought/Action/Action Input/Observation несколько раз для получения более четкого ответа)
Final Answer: (ответ пользователю)

Few-shot пример:
    Question: Что такое неустойка?
    Thought: Нужно найти определение в документе doc.docx
    Action: search_documents
    Action Input: {{"query": "определение неустойки в документе"}}
    Observation: Неустойка - денежная сумма, которую должник обязан уплатить...
    Thought: Нашел ответ в документах
    Final Answer: Неустойка - денежная сумма, которую должник обязан уплатить кредитору в случае неисполнения обязательств (ст. 330 ГК РФ).

Начало работы:
    Question: {input}
    Thought: {agent_scratchpad}

    
"""

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
        max_iterations=5,
        early_stopping_method="force"
    )
    
    return agent_executor

def chat_with_session(message: str, session_id: str = None) -> tuple:
    global current_session_id, agent_executor
    if session_id:
        current_session_id = session_id
    elif current_session_id is None:
        current_session_id = session_memory.create_session()
    

    session_memory.add_message(current_session_id, "user", message)
    context = session_memory.get_context_string(current_session_id)
    
    if context:
        full_query = f"Контекст разговора:\n{context}\n\nПользователь: {message}\nАссистент:"
    else:
        full_query = message
    
    response = agent_executor.invoke({
        "input": full_query,
        "date": date.today().strftime('%d.%m.%Y')
    })
    
    answer = response['output']
    session_memory.add_message(current_session_id, "assistant", answer)
    
    return answer, current_session_id

def new_session():
    global current_session_id
    current_session_id = session_memory.create_session()
    print(f"Новая сессия: {current_session_id}")


def list_sessions():
    sessions = session_memory.get_all_sessions()
    if not sessions:
        print("Нет сессий")
        return
    print("\nСессии:")
    for s in sessions:
        print(f"  {s['session_id']}: {s['title']} ({s['message_count']} сообщ)")


def switch_session(session_id: str):
    global current_session_id
    if session_id in session_memory._sessions:
        current_session_id = session_id
        print(f"Переключено на {session_id}")
        history = session_memory.get_history(session_id, limit=3)
        if history:
            print("\nПоследние сообщения:")
            for msg in history:
                role = "Пользователь" if msg['role'] == 'user' else "Ассистент"
                print(f"  {role}: {msg['content'][:80]}...")
    else:
        print(f"Сессия {session_id} не найдена")

def delete_session(session_id: str) -> str:
    global current_session_id
    
    if session_id not in session_memory._sessions:
        return f"Сессия {session_id} не найдена"
    
    session_memory.delete_session(session_id)
    
    if current_session_id == session_id:
        current_session_id = session_memory.create_session()
        return f"Сессия {session_id} удалена. Создана новая: {current_session_id}"
    
    return f"Сессия {session_id} удалена"

def get_current_session_id() -> str:
    global current_session_id
    return current_session_id

def set_current_session_id(session_id: str):
    global current_session_id
    if session_id in session_memory._sessions:
        current_session_id = session_id
        print(f"Синхронизирована сессия: {current_session_id}")
        return True
    return False


def get_all_sessions() -> List[Dict]:
    return session_memory.get_all_sessions()


def get_session_history(session_id: str = None) -> List[Dict]:
    if session_id is None:
        session_id = current_session_id
    return session_memory.get_history(session_id)

def safe_exit():
    os._exit(0)


def main():
    global current_session_id, agent_executor

    if not Path(MODEL_PATH).exists():
        print(f"Модель не найдена: {MODEL_PATH}")
        print("Укажите правильный путь к модели")
        return
    
    print(f"Загрузка агента...")
    agent_executor = create_agent()
    print("Агент готов!")
    
    print("\n")
    print("Интерактивный режим. Команды:")
    print("  /help - показать инструменты")
    print("  /files - список файлов")
    print("  /exit - выход")
    print("\n")
    
    running = True
    while running:
        try:
            query = input("\n > ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['/exit', '/quit', 'exit', 'quit']:
                running=False
                safe_exit()
            
            if query.lower() == '/help':
                print("  • vector_list - список файлов")
                print("  • vector_add - добавить файл в память")
                print("  • search_documents - искать информацию через RAG")
                print("  • write_file - записать данные в файл")
                print("  • execute_python - выполнить Python-код")
                continue
            
            if query.lower() == '/files':
                print(list_files())
                continue
  
            print("\nОбработка запроса...")
            answer = chat_with_session(query)
            print(f"\n{answer}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
