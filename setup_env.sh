#!/bin/bash
# O365 Search Agent 环境安装脚本
# 一键安装所有依赖，解决版本兼容问题

set -e  # 遇到错误立即停止

echo "=========================================="
echo "  O365 Search Agent 环境安装脚本"
echo "=========================================="

# 配置
ENV_NAME="cpq"
AGENT_LIGHTNING_DIR="/scratch/azureml/cr/j/d2ee4dd4d91b4ba581cca02777af6d50/cap/data-capability/wd/INPUT_src/agent-lightning"
O365_SEARCH_DIR="/scratch/azureml/cr/j/d2ee4dd4d91b4ba581cca02777af6d50/cap/data-capability/wd/INPUT_src/o365_search"

# 1. 停止 Ray（如果在运行）
echo ""
echo "[1/8] 停止 Ray..."
ray stop 2>/dev/null || true

# 2. 删除旧环境
echo ""
echo "[2/8] 删除旧环境 ${ENV_NAME}..."
conda deactivate 2>/dev/null || true
conda remove -n ${ENV_NAME} --all -y 2>/dev/null || true

# 3. 创建新环境
echo ""
echo "[3/8] 创建新环境 ${ENV_NAME} (Python 3.11)..."
conda create -n ${ENV_NAME} python=3.11 -y

# 4. 激活环境
echo ""
echo "[4/8] 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 5. 安装 PyTorch (CUDA 11.8)
echo ""
echo "[5/8] 安装 PyTorch 2.4.0 (CUDA 11.8)..."
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. 安装 vllm 和 verl
echo ""
echo "[6/8] 安装 vLLM 0.6.3 和 VERL 0.3.0..."
pip install vllm==0.6.3.post1 verl==0.3.0.post1

# 7. 安装 Flash Attention
echo ""
echo "[7/8] 安装 Flash Attention..."
pip install flash-attn --no-build-isolation

# 8. 安装 agent-lightning 和其他依赖
echo ""
echo "[8/8] 安装 agent-lightning 和其他依赖..."
cd ${AGENT_LIGHTNING_DIR}
pip install -e .
pip install wandb pyarrow pandas transformers sentence-transformers faiss-cpu

# 完成
echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
echo "接下来请执行："
echo "  1. conda activate ${ENV_NAME}"
echo "  2. ray start --head"
echo "  3. cd ${O365_SEARCH_DIR}"
echo "  4. bash train.sh"
echo ""
