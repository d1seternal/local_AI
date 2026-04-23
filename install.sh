#!/bin/bash
set -e

# Python
sudo apt update
sudo apt install -y python3.11 python3-pip python3-venv
# CMake
sudo apt install -y cmake

# Создание виртуального окружения
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac

# Установка пакетов
pip install -r requirements.txt

# Установка llama-cpp-python с оптимизацией
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --no-cache-dir