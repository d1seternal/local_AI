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
    DocumentProcessor,
    MODEL_PATH,
    DATA_DIR,
    EMBEDDING_MODEL
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
        
vector_memory = VectorMemory()

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
                stop=stop or ["Observation:", "User:", "FINAL ANSWER:", "Assistant:"]
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
        """Ты - полезный AI-ассистент для работы с документами и файлами. Сегодня {date}.

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
Thought: (результат получен, можно отвечать)
Final Answer: (ответ пользователю)

Если ты нашел ответ и готов ответить пользователю:
- FINAL ANSWER: [твой ответ] (например, "файл добавлен в память" или "Результаты поиска: <фрагменты>"), то завершай свою генерацию и ожидай новый запрос пользователя. Не нужно размышлять дальше, доходя до лимита итераций.

Никогда не пиши комментарии, заметки или пояснения, выходящие за рамки этого формата.
Никогда не пиши ничего вроде "Подожду результат..." и другие комментарии после Action Input
После Action Input идет исключительно Observation с полученным результатом

Пример работы после Action Input:
- Observation: Успешно добавлено 60 фрагментов из файла dogovor.docx в векторную базу.
- Thought: Файл успешно обработан и добавлен в векторную базу. Теперь у меня есть информация для ответа.
- FINAL ANSWER: Файл добавлен в память. Успешно добавлено 60 фрагментов из файла dogovor.docx

Начало работы:
Question: {input}
Thought: {agent_scratchpad}"""

#         """Ты - полезный AI-ассистент для работы с документами и файлами. Сегодня {date}.
# **Инструкции**:
#     -проанализируй задачу 
#     -проанлизируй историю 
#     -выбери подходящий инструмент или FINISH, если задача была выполнена. 
#     -Action: (название инструмента из списка)
#     -Action Input: (входные данные для инструмента в формате JSON)
#     -если ты уже вызывал инструмент с такими аргументами и  инструмент был выполнен успешно, то не вызывай его снова. Переходи к следующему инструменту или к FINISH.
#     -если ты выбираешь FINISH, то [твой ответ] (например, "файл добавлен в память" или "Результаты поиска: <фрагменты>"),  после этого завершай свою генерацию и ожидай новый запрос пользователя""

     
# **Ответ в формате JSON**:
#     {{
#         "action": "имя_инструмента",
#         "action_input": {{
#             "arg1": "значение",
#             "arg2": "значение"
#         }},
#         "though": "обоснование выбора инструмента"
#     }}
# Начало работы:
# - Question: {input}
# - Thought: {agent_scratchpad}
        
# Верни только ответ сторого в формате JSON со всеми заполненными полями. Не добавляй других слов и комментариев

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
        max_iterations=4,
        early_stopping_method="generate"
    )
    
    return agent_executor

def safe_exit():
    os._exit(0)


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
