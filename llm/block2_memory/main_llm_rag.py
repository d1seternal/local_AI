"""
Модуль для локального запуска LLM с подключенной векторной памятью на основе llama-cpp-python и ChromaDB.

"""

import os
import time
import psutil
import json
from datetime import datetime
from llama_cpp import Llama

from memory import VectorMemory
from rag import RAGSystem

RAG_PATH = ".\\rag_data"
MODEL_PATH = ".\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MEMORY_PATH = ".\chroma_db"
COLLECTION_NAME = "rag_memory"

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
    
    def print_summary(self):
        print("\n")
        print(" ИТОГОВЫЙ ОТЧЕТ БЕНЧМАРКИНГА")
        
        print(f"\nМодель: {self.metrics['model_name']}")
        print(f"Время загрузки: {self.metrics['load_time']:.2f} сек")
        print(f"Всего запросов: {self.metrics['total_queries']}")
        print(f"Всего токенов: {self.metrics['total_tokens']}")
        print(f"Общее время генерации: {self.metrics['total_time']:.2f} сек")
        
        if self.metrics['total_tokens'] > 0:
            avg_speed = self.metrics['total_tokens'] / self.metrics['total_time']
            print(f"Средняя скорость: {avg_speed:.2f} токенов/сек")
    

def index_documents_interactive(rag):
    while True:
        print("\n1. Индексировать файл")
        print("2. Индексировать текст")
        print("3. Показать список документов")
        print("4. Удалить документ")
        print("5. Вернуться в главное меню")
        
        choice = input("\nВаш выбор: ").strip()
        
        if choice == '1':
            file_path = input("Путь к файлу: ").strip()
            if os.path.exists(file_path):
                chunks = rag.index_document(file_path)
                print(f"Добавлено {chunks} чанков")
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
            
            text = "\n".join(lines)
            source = input("Название источника: ").strip() or "user_input"
            chunks = rag.index_text(text, source=source)
            print(f"Добавлено {chunks} чанков")
        
        elif choice == '3':
            docs = rag.list_documents()
            print(f"\nДокументов в базе: {len(docs)}")
            for doc in docs:
                from datetime import datetime
                dt = datetime.fromtimestamp(doc['timestamp'])
                print(f"  {doc['doc_id']}: {doc['filename']} - {doc['chunks']} чанков ")
        
        elif choice == '4':
            doc_id = input("ID документа для удаления: ").strip()
            deleted = rag.delete_document(doc_id)
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
        temperature=1.0,                   
        top_p=0.9,
        chat_format="mistral-instruct"    #можно использовать небходимый формат для конкретной модели (например, chatml для Phi-3-mini-4k-Q4)                
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
    print(f"Контекст: {benchmark.metrics['model_info']['n_ctx']} токенов")
    print(f"Потоков: {benchmark.metrics['model_info']['n_threads']}")
    
    return llm

def generate_with_rag(llm, query_text, benchmark, memory, rag, session_id="default"):
    doc_context = rag.get_context_for_query(query_text, k=3)
    
    chat_context = memory.get_relevant_context(
        query=query_text,
        k=3,
        session_id=session_id
    )
    
    parts = []
    
    if doc_context:
        parts.append(doc_context)
        print(f"\nНайдена информация из документов")
    
    if chat_context:
        parts.append(chat_context)
    
    if parts:
        prompt = "\n\n".join(parts) + f"\n\nUser: {query_text}\nAssistant:"
    else:
        prompt = f"User: {query_text}\nAssistant:"
    
    start_time = time.time()

    response = llm(
        prompt, 
        max_tokens=1024, 
        temperature=1.0, 
        top_p=0.9,
        echo=False 
    )
    time_taken = time.time() - start_time
    
    answer = response['choices'][0]['text'].strip()
    
    token_count = response['usage']['completion_tokens']
    tokens_per_second = token_count / time_taken if time_taken > 0 else 0
    
    memory.add_message("user", query_text, session_id=session_id)
    memory.add_message("assistant", answer, session_id=session_id)
    benchmark.add_query_result(query_text, answer, token_count, time_taken, tokens_per_second)
    
    return answer, token_count, time_taken, tokens_per_second

def chat_loop_with_rag(llm, benchmark, memory, rag, session_id):
    print(f"Сессия: {session_id}")
    print("\nКоманды:")
    print("  /exit - выход")
    print("  /search <запрос> - поиск по документам")
    print("  /docs - список документов")
    print("  /memory - статистика")
    
    while True:
        user_input = input("\nВы: ").strip()
        
        if user_input.lower() in ['/exit', 'exit', 'quit']:
            break
        
        if user_input.startswith('/search'):
            query = user_input[7:].strip()
            if query:
                results = rag.search(query, k=5)
                print(f"\nНайдено {len(results)} результатов:")
                for i, r in enumerate(results, 1):
                    source = r['metadata'].get('filename', 'unknown')
                    print(f"\n{i}. [релевантность: {r['relevance_score']:.2f}, из: {source}]")
                    print(r['text'])
            continue
        
        if user_input == '/docs':
            docs = rag.list_documents()
            print(f"\nДокументов в базе: {len(docs)}")
            for doc in docs:
                print(f"  {doc['filename']}: {doc['chunks']} чанков")
            continue
        
        if user_input == '/memory':
            print(f"\nДиалогов: {memory.count()}")
            print(f"Чанков документов: {rag.count()}")
            continue
        
        if not user_input:
            continue

        print("\nАссистент: ", end="", flush=True)
        try:
            answer, tokens, time_taken, speed = generate_with_rag(
                llm, query_text=user_input, benchmark=benchmark, memory=memory, rag=rag, session_id=session_id
            )
            print(answer)
            print(f"\n[{tokens} токенов | {time_taken:.2f} сек | {speed:.1f} ток/сек]")
        except Exception as e:
            print(f"\nОшибка: {e}")

def main():

    print("\nИнициализация векторной памяти...")
    memory = VectorMemory(
        persist_directory=MEMORY_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print(f"Память создана, папка: {MEMORY_PATH}")

    rag = RAGSystem(
        persist_directory=RAG_PATH,
        collection_name="documents",
        chunk_size=500,
        chunk_overlap=50
    )
    
    print(f"   Диалогов в памяти: {memory.count()}")
    print(f"   Чанков документов: {rag.count()}")
    
    print("\n")
    print(" ВЫБОР РЕЖИМА РАБОТЫ\n")
    print("1. Чат с ассистентом")
    print("2. Управление документами")
    print("3. Тестовые запросы")

    choice = input("\nВаш выбор: ").strip()
    
    if choice == '2':
        index_documents_interactive(rag)
        return
    
    benchmark = ModelBenchmark()
    try:
        llm = load_model(benchmark)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    if choice == '3':
        test_queries = [
        "Объясни принцип квантизации llm-моделей.",
        "Напиши рецепт классического русского борща.",
        "Сколько будет 15 * 37?",
        "Что такое градиентный спуск в машинном обучении?",
        "Опиши структуру судов в России и виды судебных разбирательств, которые расмматривает суд на каждом из уровней иерархии."
        ]

        session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for query in test_queries:
            answer, _, _, _ = generate_with_rag(
                llm, query, benchmark, memory, rag, session_id
            )
            print(f"\nQ: {query}\nA: {answer[:200]}...")
    
    else:
        session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        chat_loop_with_rag(llm, benchmark, memory, rag, session_id)

if __name__ == "__main__":
    main()