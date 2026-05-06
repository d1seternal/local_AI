"""
Расширенный бенчмарк для сравнения квантизаций
"""

import os
import ast
import re
import sys
import time
import json
import yaml
import psutil
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import gc
import ctypes

from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.__init__ import ModelBenchmark

@dataclass
class QuantizationTestResult:
    quantization: str
    model_path: str
    model_name: str
    model_size_gb: float
    load_time_sec: float
    model_ram_mb: float        
    model_ram_gb: float
    system_ram_before_mb: float
    system_ram_after_mb: float
    total_tokens: int = 0
    total_time_sec: float = 0.0
    avg_speed_tokens_per_sec: float = 0.0
    queries_count: int = 0
    error_count: int = 0
    avg_quality: float = 0.0
    timestamp: str = ""


class QuantizationBenchmark:
    def __init__(self, config_path: Optional[str] = None, single_model_mode: bool = False):
        self.single_model_mode = single_model_mode
        self.config = self._load_config(config_path)
        self.models_config = self.config.get('models', {})
        self.results: Dict[str, QuantizationTestResult] = {}
        self._setup_logging()
        self.test_queries = self.config.get('test_queries', [])
        self.test_params = self.config.get('test_params', {})
        self.logger = logging.getLogger(__name__)
    
    def _resolve_path(self, path: str) -> Path:
        if not path:
            return None
        
        p = Path(path)
        if p.is_absolute() or p.exists():
            return p
        
        base_dirs = [
            Path(__file__).parent,                  
            Path(__file__).parent.parent,            
            Path(__file__).parent.parent.parent,       
            Path(__file__).parent.parent.parent.parent,
        ]
        
        for base in base_dirs:
            full_path = base / path
            if full_path.exists():
                return full_path
    
        return p
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        config_data = {}

        if config_path:
            config_file = self._resolve_path(config_path)
            if config_file and config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
        
        if not config_data:
            search_paths = [
                Path(__file__).parent / "config.yaml",
                Path(__file__).parent.parent / "config.yaml",
                Path(__file__).parent.parent.parent / "config.yaml",
            ]
            for path in search_paths:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        break
  
        if 'models' in config_data:
            resolved_models = {}
            for name, path in config_data['models'].items():
                resolved = self._resolve_path(path)
                if resolved and resolved.exists():
                    resolved_models[name] = str(resolved)
                else:
                    resolved_models[name] = path
            config_data['models'] = resolved_models
        
        return config_data
    
    def _setup_logging(self):
        output_config = self.config.get('output', {})
        logs_dir = self._resolve_path(output_config.get('logs_dir', './logs'))
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"benchmark_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def test_single_quantization(self, quant_name: str, model_path: str) -> Optional[QuantizationTestResult]:
        model_file = self._resolve_path(model_path)
        if not model_file or not model_file.exists():
            self.logger.error(f"Модель не найдена: {model_path}")
            return None
        
        self.logger.info(f"\n")
        self.logger.info(f"Тестирование: {quant_name}")
        
        benchmark = ModelBenchmark(model_file)
        
        mem_before = benchmark.get_memory_usage()
        system_ram_before = psutil.virtual_memory().used / 1024 / 1024 / 1024
        
        self.logger.info(f"До загрузки:")
        self.logger.info(f"   RAM процесса: {mem_before.get('rss', 0):.1f} GB")
        self.logger.info(f"   RAM системы: {system_ram_before:.1f} GB")
        
        model_size_gb = model_file.stat().st_size / 1024 / 1024 / 1024
        
        self.logger.info(f"Загрузка модели {quant_name}...")
        start_time = time.time()
    
        llm = Llama(
            model_path=str(model_file),
            n_ctx=self.test_params.get('n_ctx'),
            n_threads=self.test_params.get('n_threads'),
            n_gpu_layers=self.test_params.get('n_gpu_layers'),
            temperature=self.test_params.get('temperature'),
            n_batch=self.test_params.get('n_batch'),
            top_p=self.test_params.get('top_p'),
            seed=self.test_params.get('seed'),
            verbose=False
        )
        
        load_time = time.time() - start_time
        
        mem_after = benchmark.get_memory_usage()
        system_ram_after = psutil.virtual_memory().used / 1024 / 1024 / 1024
        model_info = benchmark.get_model_info(llm)
        benchmark.set_load_metrics(load_time, mem_before, mem_after, model_info)
        
        ram_usage = benchmark.calculate_model_ram_usage()
        
        self.logger.info(f"Модель загружена за {load_time:.2f} сек")
        self.logger.info(f"Использование RAM:")
        self.logger.info(f"   Модель в RAM: {ram_usage['model_ram_gb']:.2f} GB")
        self.logger.info(f"   Размер файла: {ram_usage['file_size_gb']:.2f} GB")
        
        result = QuantizationTestResult(
            quantization=quant_name,
            model_path=str(model_file),
            model_name=model_file.name,
            model_size_gb=model_size_gb,
            load_time_sec=load_time,
            model_ram_mb=ram_usage['model_ram_mb'],
            model_ram_gb=ram_usage['model_ram_gb'],
            system_ram_before_mb=system_ram_before,
            system_ram_after_mb=system_ram_after,
            timestamp=datetime.now().isoformat()
        )

        self.logger.info(f"Выполнение тестов...")
        
        syntax_result = self._test_syntax(llm)
        result.syntax_test_passed = syntax_result['passed']
        result.syntax_test_total = syntax_result['total']
        result.total_tokens += syntax_result['total_tokens']
        result.total_time_sec += syntax_result['total_time']
        
       
        instruction_result = self._test_instruction(llm)
        result.instruction_test_passed = instruction_result['passed']
        result.instruction_test_total = instruction_result['total']
        result.total_tokens += instruction_result['total_tokens']
        result.total_time_sec += instruction_result['total_time']
        

        result.queries_count = syntax_result['total'] + instruction_result['total']
        
        if result.total_tokens > 0 and result.total_time_sec > 0:
            result.avg_speed_tokens_per_sec = result.total_tokens / result.total_time_sec
        
        syntax_score = result.syntax_test_passed / result.syntax_test_total if result.syntax_test_total > 0 else 0
        instruction_score = result.instruction_test_passed / result.instruction_test_total if result.instruction_test_total > 0 else 0
        result.avg_quality = (syntax_score + instruction_score) / 2
 
        self.logger.info(f" Итоги для {quant_name}")
        self.logger.info(f"    RAM под модель: {result.model_ram_gb:.2f} GB")
        self.logger.info(f"    Время загрузки: {result.load_time_sec:.2f} сек")
        self.logger.info(f"    Средняя скорость: {result.avg_speed_tokens_per_sec:.1f} ток/сек")
        self.logger.info(f"    Синтаксис: {result.syntax_test_passed}/{result.syntax_test_total} ({syntax_score*100:.0f}%)")
        self.logger.info(f"    Инструкции: {result.instruction_test_passed}/{result.instruction_test_total} ({instruction_score*100:.0f}%)")
        self.logger.info(f"    Общее качество: {result.avg_quality:.1%}")
        
        return result
    
    def _test_syntax(self, llm) -> Dict:
        syntax_tests = [
            {
                'name': 'Сортировка списка',
                'prompt': 'Напиши функцию на Python для сортировки списка по возрастанию'
            },
            {
                'name': 'Факториал',
                'prompt': 'Напиши рекурсивную функцию на Python для вычисления факториала числа'
            },
            {
                'name': 'Класс с методом',
                'prompt': 'Создай класс Person на Python с методами __init__ и __str__'
            }
        ]
        
        passed = 0
        total = len(syntax_tests)
        total_tokens = 0
        total_time = 0
        details = []
        
        self.logger.info(f"Тесты на синтаксическую корректность:")
        
        for test in syntax_tests:
            self.logger.info(f"   {test['name']}...")
            
            try:
                start_time = time.time()
                
                response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "Ты - эксперт по Python. Сгенерируй только код без объяснений."},
                        {"role": "user", "content": test['prompt']}
                    ],
                    max_tokens=512
                )
                
                elapsed = time.time() - start_time
                answer = response['choices'][0]['message']['content']
                tokens = response['usage']['completion_tokens']
                speed = tokens / elapsed if elapsed > 0 else 0
                
                total_tokens += tokens
                total_time += elapsed
                
                code = self._extract_code(answer)
                is_valid, error = self._check_syntax(code)
                
                if is_valid:
                    passed += 1
                    self.logger.info(f"Success: {tokens} токенов, {speed:.1f} ток/с")
                else:
                    self.logger.info(f"Error: {error[:50]} ({tokens} токенов, {speed:.1f} ток/с)")
                
                details.append({
                    'name': test['name'],
                    'passed': is_valid,
                    'tokens': tokens,
                    'time': elapsed,
                    'speed': speed,
                    'error': error if not is_valid else None
                })
                
            except Exception as e:
                self.logger.info(f"Ошибка: {e}")
                details.append({'name': test['name'], 'passed': False, 'error': str(e)})
            
            time.sleep(0.3)
        
        self.logger.info(f"Результат: {passed}/{total} ({passed/total*100:.0f}%)")
        
        return {
            'passed': passed,
            'total': total,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'details': details
        }


    def _test_instruction(self, llm) -> Dict:
        """
        Тест следования инструкциям с замером скорости
        """
        instruction_tests = [
            {
                'name': 'Список фруктов',
                'prompt': 'Напиши список из 3 фруктов. Каждый фрукт должен начинаться с буквы "А". Запрещено использовать запятые, пиши каждый фрукт с новой строки',
                'constraints': ['three_items', 'letter_a', 'no_commas']
            },
            {
                'name': 'Описание города',
                'prompt': 'Опиши город Санкт-Петербург в 3-5 предложениях. Без цифр и без точек в конце предложений',
                'constraints': ['min_length:100', 'no_numbers', 'no_dots']
            },
            {
                'name': 'Инструкция по чаю',
                'prompt': 'Напиши пошаговую инструкцию заваривания чая. Каждый шаг должен содержать не более 10 слов. Всего 4 шага',
                'constraints': ['four_steps', 'max_word_per_step:10']
            }
        ]
        
        passed = 0
        total = len(instruction_tests)
        total_tokens = 0
        total_time = 0
        details = []
        
        self.logger.info(f"Тесты на следование инструкциям:")
        
        for test in instruction_tests:
            self.logger.info(f"   {test['name']}...")
            
            try:
                start_time = time.time()
                
                response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "Ты - полезный ассистент. Точно следуй инструкциям."},
                        {"role": "user", "content": test['prompt']}
                    ],
                    max_tokens=512
                )
                
                elapsed = time.time() - start_time
                answer = response['choices'][0]['message']['content']
                tokens = response['usage']['completion_tokens']
                speed = tokens / elapsed if elapsed > 0 else 0
                
                total_tokens += tokens
                total_time += elapsed
              
                violations = []
                for constraint in test['constraints']:
                    if not self._check_single_constraint(answer, constraint):
                        violations.append(constraint)
                
                if len(violations) == 0:
                    passed += 1
                    self.logger.info(f"Success: {tokens} токенов, {speed:.1f} ток/с")
                else:
                    self.logger.info(f"Нарушения: {violations} ({tokens} токенов, {speed:.1f} ток/с)")
                
                details.append({
                    'name': test['name'],
                    'passed': len(violations) == 0,
                    'tokens': tokens,
                    'time': elapsed,
                    'speed': speed,
                    'violations': violations if violations else None,
                    'generated': answer[:200]
                })
                
            except Exception as e:
                self.logger.info(f"Ошибка: {e}")
                details.append({'name': test['name'], 'passed': False, 'error': str(e)})
            
            time.sleep(0.3)
        
        self.logger.info(f"Результат: {passed}/{total} ({passed/total*100:.0f}%)")
        
        return {
            'passed': passed,
            'total': total,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'details': details
        }
    
    def _extract_code(self, text: str) -> str:
        code_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        return text.strip()
    
    def _check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        if not code:
            return False, "Пустой код"
        
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка на строке {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Ошибка: {e}"
    
    def _check_single_constraint(self, text: str, constraint: str) -> bool:
        text_lower = text.lower()
        
        if constraint == "no_commas" or constraint == "без запятых":
            return ',' not in text
        
        elif constraint == "no_dots" or constraint == "без точек":
            return '.' not in text
        
        elif constraint == "three_items" or constraint == "3 пункта":
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            return len(lines) == 3
        
        elif constraint == "letter_a" or constraint == "буква А":
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            return all(l and l[0].lower() == 'а' for l in lines)
        
        elif constraint == "no_numbers" or constraint == "без цифр":
            return not re.search(r'\d', text)
        
        elif constraint.startswith("min_length:"):
            min_len = int(constraint.split(':')[1])
            return len(text) >= min_len
        
        elif constraint.startswith("max_length:"):
            max_len = int(constraint.split(':')[1])
            return len(text) <= max_len
        
        else:
            return constraint.lower() in text_lower

    
    def run_all_benchmarks(self) -> Dict[str, QuantizationTestResult]:
        self.logger.info("\n")
        self.logger.info("СРАВНИТЕЛЬНЫЙ БЕНЧМАРКИНГ КВАНТИЗАЦИЙ")
        
        for quant_name, model_path in self.models_config.items():
            if model_path and self._resolve_path(model_path).exists():
                result = self.test_single_quantization(quant_name, model_path)
                if result:
                    self.results[quant_name] = result
            else:
                self.logger.warning(f"Модель {quant_name} не найдена: {model_path}")
        
        return self.results
    
    def generate_comparison_report(self) -> str:
        if self.single_model_mode:
            if not self.results:
                return "Нет данных"
            
            r = list(self.results.values())[0]
            lines = []
            lines.append("\n" + "="*50)
            lines.append(f"РЕЗУЛЬТАТ ТЕСТИРОВАНИЯ: {r.quantization}")
            lines.append("="*50)
            lines.append(f"Модель: {r.model_name}")
            lines.append(f"Размер файла: {r.model_size_gb:.2f} GB")
            lines.append(f"RAM под модель: {r.model_ram_mb:.0f} MB ({r.model_ram_gb:.2f} GB)")
            lines.append(f"Время загрузки: {r.load_time_sec:.2f} сек")
            lines.append(f"Средняя скорость: {r.avg_speed_tokens_per_sec:.1f} ток/сек")
            lines.append(f"Качество ответов: {r.avg_quality:.1%}")
            lines.append(f"Всего запросов: {r.queries_count}")
            lines.append(f"Всего токенов: {r.total_tokens}")
            lines.append("="*50)
            return "\n".join(lines)
        
        if not self.results:
            return "Нет данных для сравнения"
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("СРАВНИТЕЛЬНЫЙ АНАЛИЗ КВАНТИЗАЦИЙ")
        lines.append("="*80)
        lines.append(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.append("\n ТАБЛИЦА СРАВНЕНИЯ")
        lines.append("-"*80)
        lines.append(f"{'Квантизация':<12} {'Размер(G)':<10} {'RAM(GB)':<12} {'Загрузка(с)':<12} {'Скорость':<12} {'Качество':<10}")
        lines.append("-"*80)
        
        for quant, r in sorted(self.results.items()):
            lines.append(
                f"{quant:<12} {r.model_size_gb:<10.2f} {r.model_ram_gb:<12.2f} "
                f"{r.load_time_sec:<12.2f} {r.avg_speed_tokens_per_sec:<12.1f} {r.avg_quality:<10.1%}"
            )
        
        lines.append("\n ДЕТАЛИЗАЦИЯ ПО ПАМЯТИ")
        lines.append("-"*80)
        lines.append(f"{'Квантизация':<12} {'RAM(MB)':<15} {'Файл(GB)':<12}")
        lines.append("-"*80)
        
        for quant, r in sorted(self.results.items()):
            expansion = r.model_ram_mb / (r.model_size_gb * 1024) if r.model_size_gb > 0 else 0
            lines.append(
                f"{quant:<12} {r.model_ram_mb:<15.1f} {r.model_size_gb:<12.2f}"
            )
        
        lines.append("\n РЕКОМЕНДАЦИИ")
        lines.append("-"*80)
        
        if self.results:
            best_mem = min(self.results.items(), key=lambda x: x[1].model_ram_mb)
            best_speed = max(self.results.items(), key=lambda x: x[1].avg_speed_tokens_per_sec)
            best_quality = max(self.results.items(), key=lambda x: x[1].avg_quality)
            
            lines.append(f" Самая экономичная (RAM): {best_mem[0]} ({best_mem[1].model_ram_mb:.0f} MB)")
            lines.append(f" Самая быстрая генерация: {best_speed[0]} ({best_speed[1].avg_speed_tokens_per_sec:.1f} ток/с)")
            lines.append(f" Самое высокое качество: {best_quality[0]} ({best_quality[1].avg_quality:.1%})")
        
        lines.append("\n" + "="*80)
        
        return "\n".join(lines)
    
    def save_results(self):
        """Сохраняет результаты"""
        output_config = self.config.get('output', {})
        results_dir = self._resolve_path(output_config.get('results_dir', './results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_config.get('save_json', True):
            json_file = results_dir / f"quantization_results_{timestamp}.json"
            results_dict = {k: asdict(v) for k, v in self.results.items()}
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            self.logger.info(f" JSON сохранён!")
        
        if output_config.get('save_report', True):
            report_file = results_dir / f"quantization_report_{timestamp}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(self.generate_comparison_report())
            self.logger.info(f" Отчёт сохранён!")
        
        return results_dir


if __name__ == "__main__":
    benchmark = QuantizationBenchmark()
    benchmark.run_all_benchmarks()
    print(benchmark.generate_comparison_report())
    benchmark.save_results()