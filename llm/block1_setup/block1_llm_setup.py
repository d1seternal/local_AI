"""
Модуль для локального запуска LLM с бенчмаркингом LLM на основе llama-cpp-python.

"""

import os
import time
import psutil
import json
from datetime import datetime
from llama_cpp import Llama



MODEL_PATH = ".\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

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
        
    def save_to_file(self, filename=None,  directory=".\benchmarks"):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        full_path = os.path.join(directory, filename)
    
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

        print(f"Результаты сохранены в {filename}")
        return full_path

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

def generate_with_benchmark(llm, messages, benchmark, query_text):
    start_time = time.time()
   
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,                
        temperature=1.0,                  
        top_p=0.9                                    
    )
    time_taken = time.time() - start_time

    answer = response['choices'][0]['message']['content'].strip()
    
    token_count = response['usage']['completion_tokens']
    tokens_per_second = token_count / time_taken if time_taken > 0 else 0
    
    benchmark.add_query_result(query_text, answer, token_count, time_taken, tokens_per_second)
    
    return answer, token_count, time_taken, tokens_per_second

def chat_loop(llm, benchmark):
    print("\n")
    print("Введите 'exit' для выхода, 'stats' для просмотра статистики")
    print("\n")

    messages = [{"role": "system", "content": "Ты - полезный и умный ИИ-ассистент, отвечай грамотно на русском языке."}]

    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.lower() == 'stats':
            benchmark.print_summary()
            continue
        
        if not user_input.strip():
            continue
        
        messages.append({"role": "user", "content": user_input})
        print("\nАссистент: ", end="", flush=True)
        
        try:
            answer, tokens, time_taken, speed = generate_with_benchmark(llm, messages, benchmark, user_input)
            print(answer)
            print(f"\n[{tokens} токенов за {time_taken:.2f} сек, {speed:.1f} ток/сек]")

            messages.append({"role": "assistant", "content": answer})
            if len(messages) > 15:
                messages = [messages[0]] + messages[-10:]
        except Exception as e:
            print(f"\nОшибка генерации: {e}")
            messages.pop()
        
        print()

def test_single_query(llm, benchmark):
    
    test_queries = [
        "Объясни принцип квантизации llm-моделей.",
        "Напиши рецепт классического русского борща.",
        "Сколько будет 15 * 37?",
        "Что такое градиентный спуск в машинном обучении?",
        "Опиши структуру судов в России и виды судебных разбирательств, которые расмматривает суд на каждом из уровней иерархии."
    ]
    
    print("\nЗапуск тестовых запросов...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Вопрос: {query}")

        messages = [
            {"role": "system", "content": "Ты полезный и умный ассистент, отвечай на необходимые вопросы на русском языке."},
            {"role": "user", "content": query}
        ]
        
        try:
            answer, tokens, time_taken, speed = generate_with_benchmark(llm, messages, benchmark, query)
            print(f"Ответ: {answer}")
            print(f"\nСтатистика: {tokens} токенов | {time_taken:.2f} сек | {speed:.1f} ток/сек")
        except Exception as e:
            print(f"Ошибка: {e}")
        
    benchmark.print_summary()

def main():
    benchmark = ModelBenchmark()

    try:
        llm = load_model(benchmark)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    print("\nВыберите режим работы:")
    print("1. Интерактивный чат")
    print("2. Тестовые запросы")
    print("3. Сохранить результаты в файл")
    
    choice = input("Ваш выбор (1/2/3): ").strip()
    
    if choice == "2":
        test_single_query(llm, benchmark)
    elif choice == "3":
        test_single_query(llm, benchmark)
        benchmark.save_to_file()
    else:
        chat_loop(llm, benchmark)
 
    if choice != "3":
        save = input("\nСохранить результаты в файл? (y/n): ").strip().lower()
        if save == 'y':
            benchmark.save_to_file()

if __name__ == "__main__":
    main()