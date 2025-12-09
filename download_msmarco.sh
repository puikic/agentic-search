#!/bin/bash
# MS MARCO Passage Ranking 数据集下载脚本
# 用于支持 NDCG 评估的检索训练

set -e

# 创建数据目录
mkdir -p data
cd data

echo "=========================================="
echo "下载 MS MARCO Passage Ranking 数据集"
echo "=========================================="

# 1. 下载 passage collection (语料库) - 约 1GB 压缩包
echo "[1/4] 下载 passage collection (8.8M passages)..."
if [ ! -f "collection.tar.gz" ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
fi
if [ ! -f "collection.tsv" ]; then
    tar -xzf collection.tar.gz
fi
echo "✓ collection.tsv 已就绪"

# 2. 下载 queries (查询) - 约 42MB
echo "[2/4] 下载 queries..."
if [ ! -f "queries.tar.gz" ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
fi
if [ ! -f "queries.train.tsv" ]; then
    tar -xzf queries.tar.gz
fi
echo "✓ queries.train.tsv / queries.dev.tsv 已就绪"

# 3. 下载 qrels (相关性标注) - 用于计算 NDCG
echo "[3/4] 下载 qrels (相关性标注)..."
if [ ! -f "qrels.train.tsv" ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv
fi
if [ ! -f "qrels.dev.tsv" ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv
fi
echo "✓ qrels.train.tsv / qrels.dev.tsv 已就绪"

# 4. (可选) 下载 top1000 dev 用于 re-ranking 场景
echo "[4/4] 下载 top1000.dev (可选，用于 re-ranking)..."
if [ ! -f "top1000.dev.tar.gz" ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz
fi
# 注意：top1000.dev.tar.gz 解压后约 2.5GB，按需解压
# tar -xzf top1000.dev.tar.gz

echo ""
echo "=========================================="
echo "下载完成！文件说明："
echo "=========================================="
echo ""
echo "collection.tsv    - 8,841,823 passages (语料库)"
echo "                    格式: pid \\t passage_text"
echo ""
echo "queries.train.tsv - 训练查询"
echo "queries.dev.tsv   - 开发集查询"
echo "                    格式: qid \\t query_text"
echo ""
echo "qrels.train.tsv   - 训练集相关性标注 (532,761 条)"
echo "qrels.dev.tsv     - 开发集相关性标注 (59,273 条)"
echo "                    格式: qid \\t 0 \\t pid \\t relevance"
echo ""
echo "=========================================="
echo "下一步: 运行 python prepare_data.py 处理数据"
echo "=========================================="
