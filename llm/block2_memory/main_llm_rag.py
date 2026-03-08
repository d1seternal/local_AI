"""
Модуль для локального запуска LLM с подключенной векторной памятью на основе llama-cpp-python и ChromaDB.

"""

import os
import time
from typing import Dict, Optional
import psutil
import json
from datetime import datetime
from llama_cpp import Llama

from memory import VectorMemory 
from reranker import LocalLLMReranker 
from document_parser import DocumentProcessor
from prompts import DocumentSearchPrompt, EnhancedNameSearchPrompt, NumericValueSearchPrompt, KeywordSearchPrompt, UnifiedRAGPrompt


MODEL_PATH = "./models/Phi-3-mini-4k-instruct-q4.gguf"
MEMORY_PATH = "./rag_data"
MEMORY_COLLECTION = "conversations"  
DOCS_COLLECTION = "documents"     

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
    print("Загрузка GGUF модели...")
    
    mem_before = benchmark.get_memory_usage()
    start_time = time.time()
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,                  
        n_threads=8,                   
        n_gpu_layers=0,                  
        verbose=False,                    
        seed=42,                          
        temperature=0.5,                   
        top_p=0.9,
        chat_format="chatml"    # можно использовать необходимый формат для конкретной модели                
    )
    
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

def generate_with_memory(llm, query_text, benchmark, memory, session_id="default", 
                        use_docs=True, reranker=None):
    
    
    doc_texts = []
    if use_docs:
        if reranker:
            results = memory.search_with_rerank(query_text, reranker, initial_k=10, final_k=2)
        else:
            results = memory.search_documents(query_text, k=2)
        
        for r in results:
            text = r['text'].strip()
            if text and len(text) > 0:
                doc_texts.append(text)
    
    if doc_texts:
        combined_docs = " ".join(doc_texts[:2])
        if len(combined_docs) > 1000:
            combined_docs = combined_docs[:1000] + "..."
        
        prompt = f"""Ты - полезный ассистент. Отвечай на вопросы пользователя, используя предоставленную информацию.
        Информация для ответа: {combined_docs}
        Вопрос: {query_text}
        Ответ:"""
    else:
        prompt = f"""Ты - полезный ассистент. Отвечай на вопросы пользователя. 
        Вопрос: {query_text}
        Ответ:"""

    start_time = time.time()
    response = llm(
        prompt, 
        max_tokens=500, 
        temperature=0.5, 
        top_p=0.9, 
        echo=False,
        stop=["Вопрос:", "User:", "\n\n"] 
    )
    time_taken = time.time() - start_time
    
    answer = response['choices'][0]['text'].strip()
    answer = clean_model_output(answer)
    
    token_count = response['usage']['completion_tokens']
    tokens_per_second = token_count / time_taken if time_taken > 0 else 0

    if session_id:
        memory.add_message("user", query_text, session_id=session_id)
        memory.add_message("assistant", answer, session_id=session_id)
    
    benchmark.add_query_result(query_text, answer, token_count, time_taken, tokens_per_second)
    
    return answer, token_count, time_taken, tokens_per_second

def clean_model_output(text: str) -> str:
    artifacts = [
        "Assistant:", "Ассистент:", "User:", "Пользователь:",
        "[Информация", "Из документа", "[/Информация]",
        "контекст:", "Дополнительная информация:", "Ответ:"
    ]
    
    for artifact in artifacts:
        if artifact in text:
            if text.startswith(artifact):
                text = text[len(artifact):].strip()
  
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)
    
    return text

def chat_loop_with_return(llm, benchmark, memory, session_id, reranker=None):
    token_count = 0
    MAX_TOKENS_PER_SESSION = 4096
    use_docs_context = True
    
    print(f"\nСессия: {session_id}")

    
    print("\nКоманды:")
    print("  /exit - выход в главное меню")
    print("  /new - начать новую сессию")
    print("  /search <запрос> - поиск по документам")
    print("  /docs - список документов")
    print("  /tokens - показать число использованных токенов")
    print("  /memory - статистика памяти")
    while True:
        if token_count >= MAX_TOKENS_PER_SESSION:
            print(f"\nДостигнут лимит токенов ({MAX_TOKENS_PER_SESSION}) в этой сессии!")
            print("1. Начать новую сессию")
            print("2. Вернуться в главное меню")
            print("3. Продолжить текущую сессию (не рекомендуется)")
            
            choice = input("Ваш выбор: ").strip()
            if choice == '1':
                return "new_session"
            elif choice == '2':
                return "menu"
            else:
                token_count = 0  
                print("Продолжаем текущую сессию")
        
        user_input = input("\nВы: ").strip()
        
        if user_input.lower() in ['/exit', 'exit', 'quit']:
            return "menu"
        
        if user_input.lower() == '/new':
            return "new_session"
        
        if user_input.startswith('/search'):
            query = user_input[7:].strip()
            if query:
                results = memory.search_documents(query, k=3)
                print(f"\nНайдено {len(results)} результатов:")
                for i, r in enumerate(results, 1):
                    source = r['metadata'].get('filename', r['metadata'].get('source', 'unknown'))
                    print(f"\n{i}. [релевантность: {r['relevance_score']:.2f}, из: {source}]")
                    print(r['text'][:750] + "..." if len(r['text']) > 750 else r['text'])
            continue
        
        if user_input == '/docs':
            docs = memory.list_documents()
            print(f"\nДокументов в базе: {len(docs)}")
            for doc in docs:
                from datetime import datetime
                dt = datetime.fromtimestamp(doc['timestamp'])
                print(f"  {doc['filename']}: {doc['chunks']} чанков")
            continue
        
        if user_input == '/tokens':
            print(f"\nИспользовано токенов в сессии: {token_count}")
            print(f"Лимит: {MAX_TOKENS_PER_SESSION}")
            continue
        
        if user_input == '/memory':
            stats = memory.get_stats()
            print(f"\n Статистика памяти:")
            print(f"   Диалогов: {stats['memory_messages']}")
            print(f"   Чанков документов: {stats['document_chunks']}")
            print(f"   Документов: {stats['documents']}")
            continue
        
        if not user_input:
            continue

        print("\nАссистент: ", end="", flush=True)
        try:
            answer, tokens, time_taken, speed = generate_with_memory(
                llm, 
                query_text=user_input, 
                benchmark=benchmark, 
                memory=memory, 
                session_id=session_id,
                use_docs=use_docs_context,
                reranker=reranker if use_docs_context else None
            )
            print(answer)
            token_count += tokens 
                
            print(f" | {tokens} токенов | всего: {token_count} | {time_taken:.2f} сек | {speed:.1f} ток/сек]")
            
        except Exception as e:
            print(f"\nОшибка: {e}")

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

def generate_with_document(llm, query_text, benchmark, memory, doc_id, session_id="default"):
    doc_context = memory.get_document_only_context(query_text, doc_id, k=2)
    if doc_context:
        prompt = f"""Ты - ассистент для работы с документом. Отвечай на вопросы, используя только информацию из предоставленного документа. 
        Содержание документа:{doc_context[:1000]}{'...' if len(doc_context) > 1000 else ''}
        Вопрос: {query_text}
        Ответ:"""
    else:
        prompt = f"""Ты - ассистент для работы с документом. Если информация отсутствует в документе, так и скажи. 
        Вопрос: {query_text}
        Ответ: Информация по этому вопросу отсутствует в документе."""

    start_time = time.time()
    response = llm(
        prompt, 
        max_tokens=300,  
        temperature=0.5,
        top_p=0.9, 
        echo=False,
        stop=["Вопрос:", "User:", "\n\n"]
    )
    time_taken = time.time() - start_time
    
    answer = response['choices'][0]['text'].strip()

    answer = clean_document_output(answer)
    
    token_count = response['usage']['completion_tokens']
    tokens_per_second = token_count / time_taken if time_taken > 0 else 0

    if session_id:
        memory.add_message("user", f"[Документ: {doc_id}] {query_text}", session_id=session_id)
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
    MAX_TOKENS = 4096
    
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
        embedding_model="intfloat/multilingual-e5-base",
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
            
if __name__ == "__main__":
    main()