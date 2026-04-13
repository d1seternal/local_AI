import os
import time
import psutil
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class ModelBenchmark:
    
    def __init__(self, model_path: Optional[Path] = None):
        self.metrics = {
            'model_name': os.path.basename(str(model_path)) if model_path else 'unknown',
            'model_size_gb': 0,
            'load_time': 0,
            'total_queries': 0,
            'total_tokens': 0,
            'total_time': 0,
            'queries': [],
            'memory': {
                'before_load': {},
                'after_load': {}
            },
            'system_info': {
                'cpu_cores': psutil.cpu_count(),
                'ram_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024 }
        }

        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        mem = self.process.memory_info()
        return {
            'rss': mem.rss / 1024 / 1024,  
            'vms': mem.vms / 1024 / 1024  
        }
    
    def get_model_info(self, llm) -> Dict[str, Any]:
        return {
            'n_ctx': llm.context_params.n_ctx,
            'n_threads': llm.context_params.n_threads,
            'model_size': os.path.getsize(llm.model_path) / 1024 / 1024 / 1024 if llm.model_path else 0
        }
    
    def add_query_result(self, query: str, response: str, tokens: int, 
                         time_taken: float, tokens_per_second: float):
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
    
    def set_load_metrics(self, load_time: float, mem_before: Dict, mem_after: Dict, 
                         model_info: Dict):
        self.metrics['load_time'] = load_time
        self.metrics['memory']['before_load']  = mem_before
        self.metrics['memory']['after_load']= mem_after
        self.metrics['model_info'] = model_info
        self.metrics['model_size_gb'] = model_info.get('model_size', 0)
    
    def calculate_model_ram_usage(self) -> Dict[str, Any]:
        before = self.metrics.get('memory', {}).get('before_load', {}).get('rss', 0)
        after = self.metrics.get('memory', {}).get('after_load', {}).get('rss', 0)
        model_ram = after - before
        
        file_size = self.metrics.get('model_size_gb', 0) * 1024
        
        return {
            'model_ram_mb': round(model_ram, 2),
            'model_ram_gb': round(model_ram / 1024, 2),
            'file_size_mb': round(file_size, 2),
            'file_size_gb': round(file_size / 1024, 2),
            'ram_vs_file_ratio': round(model_ram / file_size if file_size > 0 else 0, 2),
            'before_load_mb': round(before, 2),
            'before_load_gb': round(before / 1024, 2),
            'after_load_mb': round(after, 2),
            'after_load_gb': round(after / 1024, 2)
        }
    
    def print_summary(self):
        print("\n")
        print("ИТОГОВЫЙ ОТЧЕТ БЕНЧМАРКИНГА")
        print("\n")
        
        print(f"\nМодель: {self.metrics['model_name']}")
        print(f"Время загрузки: {self.metrics['load_time']:.2f} сек")
        print(f"Всего запросов: {self.metrics['total_queries']}")
        print(f"Всего токенов: {self.metrics['total_tokens']}")
        print(f"Общее время генерации: {self.metrics['total_time']:.2f} сек")
        
        if self.metrics['total_tokens'] > 0 and self.metrics['total_time'] > 0:
            avg_speed = self.metrics['total_tokens'] / self.metrics['total_time']
            print(f"Средняя скорость: {avg_speed:.2f} токенов/сек")
        
        ram_usage = self.calculate_model_ram_usage()
        print(f"\nИспользование памяти:")
        print(f"   До загрузки: {ram_usage['before_load_mb']:.1f} MB ({ram_usage['before_load_gb']:.2f} GB)")
        print(f"   После загрузки: {ram_usage['after_load_mb']:.1f} MB ({ram_usage['after_load_gb']:.2f} GB)")
        print(f"   Модель в RAM: {ram_usage['model_ram_mb']:.1f} MB ({ram_usage['model_ram_gb']:.2f} GB)")
        print(f"   Размер файла модели: {ram_usage['file_size_mb']:.1f} MB ({ram_usage['file_size_gb']:.2f} GB)")
        print("\n")
    
    def save_to_file(self, filename: Optional[str] = None, directory: str = "./benchmarks") -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.metrics['model_name'].replace('.gguf', '')
            filename = f"benchmark_{model_name}_{timestamp}.json"
        
        full_path = os.path.join(directory, filename)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        
        print(f"Результаты сохранены в {full_path}")
        return full_path
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_queries': self.metrics['total_queries'],
            'total_tokens': self.metrics['total_tokens'],
            'total_time': self.metrics['total_time'],
            'avg_speed': self.metrics['total_tokens'] / self.metrics['total_time'] 
                        if self.metrics['total_time'] > 0 else 0,
            'load_time': self.metrics['load_time'],
            'model_name': self.metrics['model_name']
        }


__all__ = ['ModelBenchmark']