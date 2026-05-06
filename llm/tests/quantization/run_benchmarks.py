#!/usr/bin/env python3
"""
Скрипт для запуска бенчмаркинга квантизаций
Использование:
    python run_benchmarks.py --config config.yaml
    python run_benchmarks.py --q3 --q4 --q5
    python run_benchmarks.py --q4
"""

import argparse
import sys
import yaml
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.quantization.quantization_benchmark import QuantizationBenchmark


def find_config():
    search_paths = [
        Path(__file__).parent / "config.yaml",         
        Path(__file__).parent.parent / "config.yaml",    
        Path(__file__).parent.parent.parent / "config.yaml",  
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Бенчмаркинг квантизаций LLM")
    
    parser.add_argument("--config", type=str, help="Путь к YAML конфигу")
    parser.add_argument("--q4", action="store_true", help="Тестировать Q4_K_M модель")
    parser.add_argument("--q5", action="store_true", help="Тестировать Q5_K_M модель")
    parser.add_argument("--q3", action="store_true", help="Тестировать Q3_K модель")
    parser.add_argument("--all", action="store_true", help="Тестировать все модели из конфига")
    parser.add_argument("--output", type=str, help="Директория для результатов")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результаты")
    
    args = parser.parse_args()
    
    config_path = args.config
    if not config_path:
        config_path = find_config()
        if not config_path:
            print("Не найден config.yaml")
            sys.exit(1)
    
    if args.q4 or args.q5 or args.q3 or args.all:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        all_models = config.get('models', {})
        models_to_test = {}
        
        if args.q4 and 'Q4_K_M' in all_models:
            models_to_test['Q4_K_M'] = all_models['Q4_K_M']
        if args.q5 and 'Q5_K_M' in all_models:
            models_to_test['Q5_K_M'] = all_models['Q5_K_M']
        if args.q3 and 'Q3_K_M' in all_models:
            models_to_test['Q3_K_M'] = all_models['Q3_K_M']
        if args.all:
            models_to_test = all_models
        
        if not models_to_test:
            print("Нет выбранных моделей для тестирования")
            sys.exit(1)
        
        temp_config = config.copy()
        temp_config['models'] = models_to_test
        
        temp_file = Path(tempfile.gettempdir()) / "temp_quant_config.yaml"
        with open(temp_file, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        benchmark = QuantizationBenchmark(str(temp_file), single_model_mode=(len(models_to_test) == 1))
        benchmark.run_all_benchmarks()
        print(benchmark.generate_comparison_report())
        
        if not args.no_save:
            benchmark.save_results()
        
        temp_file.unlink()
        
    else:
        benchmark = QuantizationBenchmark(config_path, single_model_mode=False)
        benchmark.run_all_benchmarks()
        print(benchmark.generate_comparison_report())
        
        if not args.no_save:
            benchmark.save_results()
    
    print("\n Бенчмаркинг завершён!")


if __name__ == "__main__":
    main()