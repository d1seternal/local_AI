"""
Модуль для локального запуска LLM с бенчмаркингом LLM на основе llama-cpp-python.

"""

import os
import time
import psutil
import json
from datetime import datetime
from llama_cpp import Llama

from shared.__init__ import (
    MODEL_PATH,
    MODEL_CONTEXT,
    MODEL_THREADS,
    MODEL_GPU_LAYERS,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    SEED,
    ModelBenchmark  
)

def load_model(benchmark):
    print("Загрузка GGUF модели...")
    
    mem_before = benchmark.get_memory_usage()
    ram_total = psutil.virtual_memory().total / 1024 / 1024 / 1024
    
    start_time = time.time()
    
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=MODEL_CONTEXT,                  
        n_threads=MODEL_THREADS,                   
        n_gpu_layers=MODEL_GPU_LAYERS,                  
        verbose=False,                    
        seed=SEED,                          
        temperature=MODEL_TEMPERATURE,                   
        top_p=MODEL_TOP_P,
        chat_format="mistral-instruct",   #можно использовать небходимый формат для конкретной модели (например, chatml для Phi-3-mini-4k-Q4)                
    )
    
    load_time = time.time() - start_time
    model_info = benchmark.get_model_info(llm)
    mem_after = benchmark.get_memory_usage()
    benchmark.set_load_metrics(load_time, mem_before, mem_after, model_info)
    
    print(f"Модель загружена: {os.path.basename(str(MODEL_PATH))}")
    print(f"Время загрузки: {load_time:.2f} сек")
    print(f"\nИспользование памяти:\n")
    print(f"До загрузки: {(mem_before['rss'] / 1024):.1f} GB")
    print(f"После загрузки: {(mem_after['rss'] / 1024):.1f} GB")
    print(f"Осталось свободно: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")
    print(f"Контекст: {benchmark.metrics['model_info']['n_ctx']} токенов")
    print(f"Потоков: {benchmark.metrics['model_info']['n_threads']}")
    
    return llm

def generate_with_benchmark(llm, messages, benchmark, query_text):
    start_time = time.time()
   
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,                
        temperature=MODEL_TEMPERATURE,                  
        top_p=MODEL_TOP_P                                  
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
            {"role": "system", "content": "Ты - полезный и умный ассистент, отвечай на необходимые вопросы на русском языке."},
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
        benchmark.print_summary()
    elif choice == "3":
        test_single_query(llm, benchmark)
        benchmark.print_summary()
        benchmark.save_to_file()
    else:
        chat_loop(llm, benchmark)
        benchmark.print_summary()
 
    if choice != "3":
        save = input("\nСохранить результаты в файл? (y/n): ").strip().lower()
        if save == 'y':
            benchmark.save_to_file()

if __name__ == "__main__":
    main()