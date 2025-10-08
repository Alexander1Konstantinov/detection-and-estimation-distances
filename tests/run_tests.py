"""
Скрипт для запуска всех тестов проекта.
"""

import unittest
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))


def run_tests() -> Tuple[unittest.TestResult, float]:
    """
    Запуск всех тестов и возврат результатов.
    
    Returns:
        Tuple[unittest.TestResult, float]: Результаты тестов и время выполнения
    """
    start_time = time.time()
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(suite)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time


def print_test_summary(result: unittest.TestResult, execution_time: float) -> None:
    """
    Печать сводки результатов тестирования.
    
    Args:
        result: Результаты тестов
        execution_time: Время выполнения
    """
    print("\n" + "="*70)
    print("ТЕСТОВАЯ СВОДКА")
    print("="*70)
    
    print(f"Всего тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Провалены: {len(result.failures)}")
    print(f"Ошибки: {len(result.errors)}")
    print(f"Время выполнения: {execution_time:.2f} секунд")
    
    if result.failures:
        print("\nПРОВАЛЕННЫЕ ТЕСТЫ:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print("\nТЕСТЫ С ОШИБКАМИ:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"\nУспешность тестов: {success_rate:.1f}%")

        
    
    print("="*70)


def main() -> int:
    """
    Основная функция запуска тестов.
    
    Returns:
        int: Код возврата (0 - успех, 1 - провал)
    """
    result, execution_time = run_tests()
    print_test_summary(result, execution_time)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)