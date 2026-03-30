"""
Модуль для локального запуска LLM с подключенной векторной памятью на основе llama-cpp-python и ChromaDB.

"""

import os
from pathlib import Path
import re
import time
from typing import Dict, Optional, List
import psutil
from datetime import datetime
from llama_cpp import Llama
from pydantic import BaseModel, Field
import argparse
import json
import sys
import traceback
import io

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.__init__ import (
    VectorMemory,
    DocumentProcessor,
    LocalLLMReranker,
    DocumentSearchPrompt,
    MODEL_PATH,
    MEMORY_PATH,
    MEMORY_COLLECTION,
    DOCS_COLLECTION,
    EMBEDDING_MODEL
)


_shared_llm = None
_shared_benchmark = None
_shared_memory = None
_shared_processor = None

def set_shared_components(llm, benchmark=None, memory=None, processor=None):
    global _shared_llm, _shared_benchmark, _shared_memory, _shared_processor
    _shared_llm = llm
    _shared_benchmark = benchmark
    _shared_memory = memory
    _shared_processor = processor
    print("[RAG] Получены общие компоненты от агента")

def get_shared_llm():
    return _shared_llm

def get_shared_benchmark():
    return _shared_benchmark

def get_shared_memory():
    return _shared_memory

def get_shared_processor():
    return _shared_processor

class ModelBenchmark:
    
    def __init__(self):
        self.metrics = {
            'model_name': os.path.basename(MODEL_PATH),
            'model_size_gb': 0,
            'load_time': 0,
            'total_queries': 0,
            'total_tokens': 0,
            'total_time': 0,
            'queries': []
        }
        self.process = psutil.Process()
    
    def get_memory_usage(self):
        mem = self.process.memory_info()
        return {
            'rss': mem.rss / 1024 / 1024, 
            'vms': mem.vms / 1024 / 1024 
        }
    
    def get_model_info(self, llm):
        info = {
            'n_ctx': llm.context_params.n_ctx,
            'n_threads': llm.context_params.n_threads,
            'model_size': os.path.getsize(MODEL_PATH) / 1024 / 1024 / 1024
        }
        return info
    
    def add_query_result(self, query, response, tokens, time_taken, tokens_per_second):
        self.metrics['queries'].append({
            'query': query,
            'response_length': len(response),
            'tokens_generated': tokens,
            'time_seconds': round(time_taken, 2),
            'tokens_per_second': round(tokens_per_second, 2)
        })
        self.metrics['total_queries'] += 1
        self.metrics['total_tokens'] += tokens
        self.metrics['total_time'] += time_taken
    
    def calculate_model_ram_usage(self):
        
        before = self.metrics['memory']['before_load']['ram_mb']
        after = self.metrics['memory']['after_load']['ram_mb']
        model_ram = after - before
        
        file_info = self.metrics.get('model_file_info', {})
        file_size = file_info.get('size_gb', 0) * 1024
        
        return {
            'model_ram_mb': round(model_ram, 2),
            'model_ram_gb': round(model_ram / 1024, 2),
            'file_size_mb': round(file_size, 2),
            'ram_vs_file_ratio': round(model_ram / file_size if file_size > 0 else 0, 2),
            'before_load_mb': round(before, 2),
            'after_load_mb': round(after, 2),
            'peak_ram_mb': self.metrics['memory']['peak'].get('ram_mb', 0)
        }
    


def index_documents_interactive(memory):
    while True:
        print("1. Индексировать файл")
        print("2. Индексировать текст")
        print("3. Показать список документов")
        print("4. Удалить документ")
        print("5. Вернуться в главное меню")
        
        choice = input("\nВаш выбор: ").strip()
        
        if choice == '1':
            file_path = input("Путь к файлу: ").strip()
            if os.path.exists(file_path):
                try:
                    chunks = memory.index_document(file_path)
                    print(f"Добавлено {chunks} чанков")
                except Exception as e:
                    print(f"Ошибка: {e}")
            else:
                print("Файл не найден")
        
        elif choice == '2':
            print("Введите текст (две пустые строки для окончания):")
            lines = []
            while True:
                line = input()
                if line == "" and (not lines or lines[-1] == ""):
                    break
                lines.append(line)
            
            if lines:
                text = "\n".join(lines)
                source = input("Название источника: ").strip() or "user_input"
                chunks = memory.index_text(text, source=source)
                print(f"Добавлено {chunks} чанков")
        
        elif choice == '3':
            docs = memory.list_documents()
            print(f"\nДокументов в базе: {len(docs)}")
            for doc in docs:
                from datetime import datetime
                dt = datetime.fromtimestamp(doc['timestamp'])
                print(f" {doc['filename']}: {doc['chunks']} чанков")
        
        elif choice == '4':
            doc_id = input("ID документа для удаления: ").strip()
            deleted = memory.delete_document(doc_id)
            if deleted:
                print(f"Удалено {deleted} чанков")
            else:
                print("Документ не найден")
        
        elif choice == '5':
            break

def load_model(benchmark):

    shared_llm = get_shared_llm()

    if shared_llm is not None:
        print("[RAG] Использую общую модель от агента", file=sys.stderr)
        return shared_llm
    
    else:
        print("Загрузка GGUF модели...")

        mem_before = benchmark.get_memory_usage()
        start_time = time.time()
        
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=32768,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False,
            seed = 42,
            temperature = 0.5,
            top_p=0.9
        )
        print("[RAG] Модель загружена")

        load_time = time.time() - start_time
        mem_after = benchmark.get_memory_usage()
        
        benchmark.metrics['load_time'] = load_time
        benchmark.metrics['memory_before'] = mem_before
        benchmark.metrics['memory_after'] = mem_after
        benchmark.metrics['model_info'] = benchmark.get_model_info(llm)
        
        print(f"Модель загружена: {os.path.basename(MODEL_PATH)}")
        print(f"Время загрузки: {load_time:.2f} сек")
        print(f"Использование памяти: {mem_after['rss']:.1f} MB")
        print(f"Потоков: {benchmark.metrics['model_info']['n_threads']}")

    return llm


def generate_with_prompts(
    llm, 
    query_text: str, 
    benchmark, 
    memory, 
    session_id: str = "default", 
    use_docs: bool = True,
    reranker = None
):
    doc_texts = []
    relevant_chunks = []
    
    keywords = extract_keywords(query_text)
    if use_docs:
        if reranker:
            results = memory.search_with_rerank(query_text, reranker, initial_k=5, final_k=2)
        else:
            results = memory.search_documents(query_text, k=2)
        
        relevant_chunks = results

        context_parts = []
        for i, r in enumerate(results, 1):
            text = r['text'].strip()
            if text:
                if len(text) > 800:
                    text = text[:800] + "..."
                context_parts.append(f"[ЧАСТЬ {i}]\n{text}")
                doc_texts.append(text)
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "Контекст пуст"
    else:
        context = "Документы не используются."
    
    system_prompt = DocumentSearchPrompt.system_prompt
    user_prompt = DocumentSearchPrompt.user_prompt.format(
        context=context,
        question=query_text,
        keywords=", ".join(keywords)
    )
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nОтвет:"

    start_time = time.time()
    
    response = llm(
        full_prompt, 
        max_tokens=500,
        temperature=0.5,
        top_p=0.9, 
        echo=False
    )
    
    time_taken = time.time() - start_time
    
    raw_answer = response['choices'][0]['text'].strip()
    raw_answer = clean_model_output(raw_answer)
    
    token_count = response['usage']['completion_tokens']
    tokens_per_second = token_count / time_taken if time_taken > 0 else 0

    return raw_answer, token_count, time_taken, tokens_per_second

def clean_model_output(text: str) -> str:
    artifacts = [
        "Assistant:", "Ассистент:", "User:", "Пользователь:",
        "[Информация", "Из документа", "[/Информация]",
        "контекст:", "Дополнительная информация:", "Ответ:",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"
    ]
    
    for artifact in artifacts:
        if artifact in text:
            if text.startswith(artifact):
                text = text[len(artifact):].strip()
            text = text.replace(artifact, "")
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)
    
    return text


def extract_keywords(question: str) -> List[str]:
    
    stop_words = {
        'какой', 'какая', 'какое', 'какие', 'кто', 'что', 'где', 
        'когда', 'почему', 'зачем', 'как', 'сколько', 'это', 'тот',
        'этот', 'весь', 'все', 'они', 'она', 'оно', 'мы', 'вы',
        'в', 'на', 'с', 'со', 'из', 'по', 'к', 'у', 'от', 'для',
        'и', 'а', 'но', 'или', 'что', 'чтобы', 'быть', 'иметь'
    }
    
    clean_question = re.sub(r'[^\w\s]', ' ', question.lower())
    words = clean_question.split()
    
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return list(set(keywords))[:5]

def chat_loop_with_return(llm, benchmark, memory, session_id, reranker=None):
    token_count = 0
    MAX_TOKENS_PER_SESSION = 8192
    use_docs_context = True
    
    print(f"\nСессия: {session_id}")
    
    print("\nКоманды:")
    print("  /exit - выход в главное меню")
    print("  /new - начать новую сессию (сбросить токены)")
    print("  /clear - очистить историю текущей сессии")
    print("  /history - показать историю текущей сессии")
    print("  /all_history - показать всю историю диалогов")
    print("  /delete_history <id> - удалить историю сессии по ID")
    print("  /delete_all_history - удалить всю историю диалогов")
    print("  /search <запрос> - поиск по документам")
    print("  /docs - список документов")
    print("  /tokens - показать число использованных токенов")
    print("  /memory - статистика памяти")
    
    while True:
        if token_count >= MAX_TOKENS_PER_SESSION:
            print(f"\nДостигнут лимит токенов ({MAX_TOKENS_PER_SESSION}) в этой сессии!")
            choice = input("Начать новую сессию? (1 - да, 2 - нет): ").strip()
            if choice == '1':
                return "new_session"
            else:
                token_count = 0
                print("Продолжаем текущую сессию")
        
        user_input = input("\nВы: ").strip()
        
        if user_input.lower() in ['/exit', 'exit', 'quit']:
            return "menu"
        
        if user_input.lower() == '/new':
            print("\nНачинаем новую сессию...")
            return "new_session"
        
        if user_input.lower() == '/clear':
            deleted = memory.delete_session(session_id)
            token_count = 0
            print(f"История текущей сессии очищена")
            continue

        if user_input.lower() == '/history':
            messages = memory.get_session_messages(session_id, limit=50)
            if messages:
                print(f"\nИстория сессии {session_id}:")
                for i, msg in enumerate(messages, 1):
                    role = msg['metadata'].get('role', 'unknown')
                    text = msg['text'].replace(f"{role}: ", "")[:100]
                    timestamp = datetime.fromtimestamp(msg['metadata'].get('timestamp', 0)).strftime('%H:%M:%S')
                    print(f"  {i}. [{timestamp}] {role}: {text}...")
                print(f"\nВсего сообщений: {len(messages)}")
            else:
                print("История текущей сессии пуста")
            continue
        
        if user_input.lower() == '/all_history':
            all_messages = memory.memory_collection.get()
            
            if all_messages['ids']:
                sessions = {}
                for i in range(len(all_messages['ids'])):
                    session = all_messages['metadatas'][i].get('session_id', 'unknown')
                    if session not in sessions:
                        sessions[session] = 0
                    sessions[session] += 1
                
                print(f"\nВся история диалогов:")
                print(f"   Всего сообщений: {len(all_messages['ids'])}")
                print(f"   Сессий: {len(sessions)}")
                print("\n Список сессий:")
                for session, count in sorted(sessions.items(), key=lambda x: x[0]):
                    preview = session
                    print(f"  • {preview}: {count} сообщений")
                print("\n Используйте /delete_history <id> для удаления конкретной сессии")
            else:
                print("История диалогов пуста")
            continue

        if user_input.startswith('/delete_history'):
            parts = user_input.split()
            if len(parts) == 2:
                target_session = parts[1].strip()
                deleted = memory.delete_session(target_session)
                if deleted > 0:
                    print(f"Сессия {target_session} удалена ({deleted} сообщений)")
                else:
                    print(f" Сессия с ID {target_session} не найдена")
            else:
                print("Использование: /delete_history <session_id>")
            continue

        if user_input.lower() == '/delete_all_history':
            confirm = input("Точно удалить всю историю диалогов? (yes/no): ").strip().lower()
            if confirm == 'yes':
                all_messages = memory.memory_collection.get()
                if all_messages['ids']:
                    memory.memory_collection.delete(ids=all_messages['ids'])
                    print(f"Вся история диалогов удалена ({len(all_messages['ids'])} сообщений)")
                else:
                    print("История и так пуста")
            else:
                print("Операция отменена")
            continue
        
        if user_input.startswith('/search'):
            query = user_input[8:].strip() if user_input.startswith('/search ') else user_input[7:].strip()
            if query:
                results = memory.search_documents(query, k=3)
                print(f"\nНайдено {len(results)} результатов:")
                for i, r in enumerate(results, 1):
                    source = r['metadata'].get('filename', r['metadata'].get('source', 'unknown'))
                    doc_id = r['metadata'].get('doc_id', 'unknown')
                    print(f"\n{i}. [релевантность: {r['relevance_score']:.2f}]")
                    print(f"   Документ: {source}")
                    print(f"   ID: {doc_id}")
                    print(r['text'][:300] + "..." if len(r['text']) > 300 else r['text'])
            continue
        
        if user_input == '/docs':
            docs = memory.list_documents()
            print(f"\nДокументов в базе: {len(docs)}")
            for i, doc in enumerate(docs, 1):
                dt = datetime.fromtimestamp(doc['timestamp']).strftime('%Y-%m-%d %H:%M')
                print(f"\n{i}. {doc['filename']}")
                print(f"   ID: {doc['doc_id']}")
                print(f"   Чанков: {doc['chunks']}")
            continue
        
        if user_input == '/tokens':
            print(f"\nИспользовано токенов в сессии: {token_count}")
            print(f"Лимит: {MAX_TOKENS_PER_SESSION}")
            continue
        
        if user_input == '/memory':
            stats = memory.get_stats()
            print(f"\n Статистика памяти:")
            print(f"   Диалогов в памяти: {stats['memory_messages']}")
            print(f"   Чанков документов: {stats['document_chunks']}")
            print(f"   Документов: {stats['documents']}")
            continue
        
        if not user_input:
            continue

        print("\nАссистент: ", end="", flush=True)
        try:
            answer, tokens, time_taken, speed = generate_with_prompts(
                llm, 
                query_text=user_input, 
                benchmark=benchmark, 
                memory=memory, 
                session_id=session_id,
                use_docs=use_docs_context,
                reranker=reranker
            )
            print(answer)
            token_count += tokens 
            
            print(f"\n[{tokens} токенов | всего: {token_count} | {time_taken:.2f} сек | {speed:.1f} ток/сек]")
            
        except Exception as e:
            print(f"\nОшибка: {e}")
            traceback.print_exc()

def chat_session(memory):
    
    chat_session.benchmark = ModelBenchmark()
    
    try:
        chat_session.llm = load_model(chat_session.benchmark)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    reranker = LocalLLMReranker(chat_session.llm, batch_size=3)
    session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = chat_loop_with_return(chat_session.llm, chat_session.benchmark, 
                                  memory, session_id, reranker)
    
    if result == "menu":
        return 
    elif result == "new_session":
        chat_session(memory)
        return

def choose_document(memory) -> Optional[Dict]:
    docs = memory.list_documents()
    if not docs:
        print("\n Пусто. Сначала загрузите документы в базу.")
        return None
    
    print("\n ДОСТУПНЫЕ ДОКУМЕНТЫ\n")
    for i, doc in enumerate(docs, 1):
        dt = datetime.fromtimestamp(doc['timestamp'])
        print(f"\n{i}. {doc['filename']}")
    
    while True:
        try:
            choice = input("Выберите документ (номер в списке): ").strip()
            if choice == '0':
                return None
            idx = int(choice)
            if 1<=idx<=len(docs):
                return docs[idx-1]
            else:
                print(f"Введите число от 0 до {len(docs)}")
        except ValueError:
            print("Введите число")

def generate_with_document(
    llm, 
    query_text: str, 
    benchmark, 
    memory, 
    doc_id: str, 
    session_id: str = "default",
    reranker = None
):
    keywords = extract_keywords(query_text)

    context_parts = []
    relevant_chunks = []

    doc_context = memory.get_document_only_context(query_text, doc_id, k=2)
    
    if doc_context:
        if reranker:
            query_embedding = memory._get_embedding(query_text, is_query=True)
            results = memory.docs_collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"doc_id": str(doc_id).strip()}
            )
            
            if results['ids'][0]:
                initial_chunks = []
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    score = max(0, 1 - distance / 2)
                    initial_chunks.append({
                        "text": results['documents'][0][i],
                        "relevance_score": score,
                        "distance": distance
                    })
                
                reranked = reranker.rerank(query_text, initial_chunks, top_k=2)
                relevant_chunks = reranked
                for i, chunk in enumerate(reranked, 1):
                    text = chunk['text'].strip()
                    if len(text) > 600: 
                        text = text[:600] + "..."
                    context_parts.append(f"[ЧАСТЬ {i+1}]\n{text}")
        else:
            query_embedding = memory._get_embedding(query_text, is_query=True)
            results = memory.docs_collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                where={"doc_id": str(doc_id).strip()}
            )
            
            if results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    text = results['documents'][0][i].strip()
                    if len(text) > 600:
                        text = text[:600] + "..."
                    context_parts.append(f"[ЧАСТЬ {i+1}]\n{text}")
                    relevant_chunks.append({
                        "text": text,
                        "relevance_score": max(0, 1 - results['distances'][0][i]/2)
                    })
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else doc_context
    else:
        context = "В документе не найдено релевантной информации."
        print("Релевантные чанки не найдены")
    
    system_prompt = DocumentSearchPrompt.system_prompt
    user_prompt = DocumentSearchPrompt.user_prompt.format(
        context=context,
        question=query_text,
        keywords=", ".join(keywords)
    )
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nОтвет:"

    start_time = time.time()
    
    response = llm(
        full_prompt, 
        max_tokens=350,
        temperature=0.5,
        top_p=0.9, 
        echo=False
    )
    
    time_taken = time.time() - start_time
    
    answer = response['choices'][0]['text'].strip()
    if not answer or len(answer) < 2:
        answer = "Информация по данному вопросу отсутствует в документе."
    
    token_count = response['usage']['completion_tokens']
    tokens_per_second = token_count / time_taken if time_taken > 0 else 0

    if session_id:
        memory.add_message("user", query_text, session_id=session_id)
        memory.add_message("assistant", answer, session_id=session_id)
    
    benchmark.add_query_result(query_text, answer, token_count, time_taken, tokens_per_second)
    
    return answer, token_count, time_taken, tokens_per_second


def clean_document_output(text: str) -> str:
    artifacts = [
        "Ответ:", "Assistant:", "Ассистент:",
        "На основе документа:", "Согласно документу:",
        "В документе сказано:", "[Документ]"
    ]
    
    for artifact in artifacts:
        if text.startswith(artifact):
            text = text[len(artifact):].strip()

    if not text or len(text) < 1:
        return "Информация по этому вопросу отсутствует в документе."
    
    return text

def chat_with_document_session(memory):
    selected_doc = choose_document(memory)
    if selected_doc is None:
        return
        
    doc_id = selected_doc['doc_id']
    filename = selected_doc['filename']
    
    print(f"\nРабота с документом: {filename}")
    
    benchmark = ModelBenchmark()
    try:
        llm = load_model(benchmark)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    session_id = f"doc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    token_count = 0
    MAX_TOKENS = 8192
    
    print(f"\n Сессия: {session_id}")
    print("\nКоманды:")
    print("  /exit - выход")
    print("  /info - информация о документе")
    print("  /clear - очистить контекст сессии")
    
    while True:
        if token_count >= MAX_TOKENS:
            print(f"\nДостигнут лимит токенов. Начинаем новую сессию.")
            session_id = f"doc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            token_count = 0
            print(f"Новая сессия: {session_id}")
        
        user_input = input("\nВы: ").strip()

        if user_input.lower() in ['/exit', 'exit', 'quit']:
            break
        
        if user_input.lower() == '/info':
            print(f"\n Информация о документе:")
            print(f"   Имя: {filename}")
            print(f"   ID: {doc_id}")
            print(f"   Чанков: {selected_doc['chunks']}")
            continue
        
        if user_input.lower() == '/clear':
            session_id = f"doc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            token_count = 0
            print("Контекст сессии очищен")
            continue
        
        if not user_input:
            continue
        
        print("\nАссистент: ", end="", flush=True)
        
        try:
            answer, tokens, time_taken, speed = generate_with_document(
                llm=llm,
                query_text=user_input,
                benchmark=benchmark,
                memory=memory,
                doc_id=doc_id,
                session_id=session_id
            )
            
            print(answer)
            token_count += tokens
            source_info = "из документа" if "отсутствует" not in answer.lower() else "информация не найдена"
            print(f"\n[{tokens} токенов | {source_info} | {time_taken:.2f} сек | {speed:.1f} ток/сек]")
            
        except Exception as e:
            print(f"\nОшибка: {e}")



# def run_cli_mode():

#     if sys.platform == 'win32':
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#         sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

#     parser = argparse.ArgumentParser(description="RAG-ассистент CLI")
#     parser.add_argument("--mode", choices=["ask", "search", "index", "list", "stats", "delete"], 
#                        required=True, help="Режим работы")
#     parser.add_argument("--query", type=str, default="", help="Поисковый запрос или вопрос")
#     parser.add_argument("--file", type=str, default="", help="Путь к файлу для индексации")
#     parser.add_argument("--doc_id", type=str, default="", help="ID документа для удаления")
#     parser.add_argument("--k", type=int, default=3, help="Количество результатов")
#     parser.add_argument("--json", action="store_true", help="Вывод в JSON формате")
    
#     args = parser.parse_args()
    
#     shared_llm = get_shared_llm()
#     shared_memory = get_shared_memory()
#     shared_processor = get_shared_processor()
    
#     if shared_memory is not None:
#         memory = shared_memory
#         processor = shared_processor
#         print("[RAG] Использую общую память от агента")
#     else:
#         processor = DocumentProcessor(
#             use_docling=True,
#             ocr_enabled=True,
#             table_mode="accurate"
#         )
        
#         memory = VectorMemory(
#             persist_directory=MEMORY_PATH,
#             memory_collection=MEMORY_COLLECTION,
#             docs_collection=DOCS_COLLECTION,
#             embedding_model="intfloat/multilingual-e5-base",
#             doc_processor=processor
#         )
    
#     if args.mode == "ask":
#         if shared_llm is not None:
#             llm = shared_llm
#             benchmark = get_shared_benchmark()
#             print("[RAG] Использую общую модель от агента")
#         else:
#             benchmark = ModelBenchmark()
#             llm = load_model(benchmark)
        
#         reranker = LocalLLMReranker(llm, batch_size=3) 
        
#         answer, tokens, time_taken, speed = generate_with_prompts(
#             llm=llm,
#             query_text=args.query,
#             benchmark=benchmark,
#             memory=memory,
#             session_id="cli_session",
#             use_docs=True,
#             reranker=reranker
#         )

#         if args.json:
#             print(json.dumps({
#                 "answer": answer,
#                 "tokens": tokens,
#                 "time": time_taken,
#                 "speed": speed
#             }, ensure_ascii=False))
#         else:
#             print(answer)
    
#     elif args.mode == "search":
#         results = memory.search_with_rerank(args.query)
        
#         if args.json:
#             json_results = []
#             for r in results:
#                 json_results.append({
#                     "text": r['text'],
#                     "relevance_score": r['relevance_score'],
#                     "metadata": r['metadata']
#                 })
#             print(json.dumps(json_results, ensure_ascii=False, default=str))
#         else:
#             for i, r in enumerate(results, 1):
#                 filename = r['metadata'].get('filename', 'unknown')
#                 score = r['relevance_score'] * 100
#                 print(f"{i}. [{score:.0f}%] {filename}")
#                 print(f"   {r['text'][:200]}...\n")


def main():
    processor = DocumentProcessor(
        use_docling=True,
        ocr_enabled=True, 
        table_mode="accurate"  
    )

    print("\nИнициализация векторной памяти...")
    memory = VectorMemory(
        persist_directory=MEMORY_PATH,
        memory_collection=MEMORY_COLLECTION,
        docs_collection=DOCS_COLLECTION,
        embedding_model=EMBEDDING_MODEL,
        doc_processor=processor
    )
    
    stats = memory.get_stats()
    print(f"   Диалогов в памяти: {stats['memory_messages']}")
    print(f"   Документов: {stats['documents']}")

    while True:
        print("\n ВЫБОР РЕЖИМА\n")
        print("1. Чат с ассистентом")
        print("2. Управление документами")
        print("3. Чат по конкретному документу")
        print("4. Выход")
        
        choice = input("\nВаш выбор: ").strip()
        
        if choice == '1':
            chat_session(memory) 
        elif choice == '2':
            index_documents_interactive(memory)
        elif choice == '3':
            chat_with_document_session(memory) 
            continue
        elif choice == '4':
            break
        else:
            print("Неверный выбор. Пожалуйста, выберите от 1 до 4")

# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] in ['--mode', '-m']:
#         run_cli_mode()
#     else:
#         main()
if __name__ == "__main__":
    main()
