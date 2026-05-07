#!/bin/bash
set -e


sudo apt install -y \
    git git-lfs \
    python3 python3-pip python3-venv \
    build-essential gcc g++ make \
    cmake pkg-config \
    libopenblas-dev \
    curl wget unzip \
    htop tmux \
    sqlite3


git lfs install

if [ ! -f /swapfile ]; then
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi


python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

pip install -r requirements.txt


CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
pip install llama-cpp-python --force-reinstall --no-cache-dir


