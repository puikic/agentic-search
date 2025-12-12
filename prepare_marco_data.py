# Copyright (c) Microsoft. All rights reserved.
"""
MS MARCO 数据处理脚本

将原始 MS MARCO 数据转换为训练格式，包含：
1. 加载 passages, queries, qrels
2. 构建训练数据（包含 relevant_passages 用于 NDCG 计算）
3. 转换 passages 为 search_r1 兼容格式
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm


def load_passages(collection_path: str) -> Dict[str, str]:
    """
    加载 passage collection
    
    格式: pid \t passage_text
    """
    print(f"Loading passages from {collection_path}...")
    passages = {}
    with open(collection_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading passages"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pid, text = parts[0], parts[1]
                passages[pid] = text
    print(f"Loaded {len(passages)} passages")
    return passages


def load_queries(queries_path: str) -> Dict[str, str]:
    """
    加载 queries
    
    格式: qid \t query_text
    """
    print(f"Loading queries from {queries_path}...")
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, text = parts[0], parts[1]
                queries[qid] = text
    print(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """
    加载 qrels (相关性标注)
    
    格式: qid \t 0 \t pid \t relevance
    返回: {qid: {pid: relevance_score}}
    """
    print(f"Loading qrels from {qrels_path}...")
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][pid] = rel
    print(f"Loaded qrels for {len(qrels)} queries")
    return dict(qrels)


def convert_to_search_r1_corpus(passages: Dict[str, str], output_path: str):
    """
    将 passages 转换为 search_r1 兼容的 JSONL 格式
    
    格式: {"docid": "xxx", "contents": "passage text"}
    """
    print(f"Converting passages to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for pid, text in tqdm(passages.items(), desc="Converting"):
            doc = {"docid": pid, "contents": text}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Saved {len(passages)} documents")


def prepare_training_data(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    output_path: str,
    max_samples: int = None
):
    """
    准备训练数据
    
    输出格式:
    {
        "question": "query text",
        "relevant_passages": {"pid1": 1, "pid2": 1, ...}  # 用于 NDCG 计算
    }
    """
    print(f"Preparing training data...")
    train_data = []
    
    for qid, query_text in tqdm(queries.items(), desc="Processing queries"):
        if qid not in qrels:
            continue
        
        relevant_passages = qrels[qid]
        
        # 只保留有相关 passage 的 query
        if not relevant_passages:
            continue
        
        sample = {
            "qid": qid,
            "question": query_text,
            # 将 dict 转为 JSON 字符串，避免 parquet 处理嵌套类型卡住
            "relevant_passages": json.dumps(relevant_passages)
        }
        train_data.append(sample)
        
        if max_samples and len(train_data) >= max_samples:
            break
    
    print(f"Saving {len(train_data)} samples to {output_path}...")
    # 保存为 parquet 格式（和 search_r1 一致）
    df = pd.DataFrame(train_data)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(train_data)} training samples to {output_path}")
    
    # 同时保存 JSONL 格式（方便查看）
    jsonl_path = output_path.replace(".parquet", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in train_data[:100]:  # 只保存前 100 条用于检查
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved sample to {jsonl_path}")


def print_data_stats(qrels: Dict[str, Dict[str, int]]):
    """打印数据统计信息"""
    total_pairs = sum(len(v) for v in qrels.values())
    avg_relevant = total_pairs / len(qrels) if qrels else 0
    
    print("\n" + "=" * 50)
    print("数据统计:")
    print("=" * 50)
    print(f"总 query 数: {len(qrels)}")
    print(f"总 query-passage 对数: {total_pairs}")
    print(f"平均每个 query 的相关 passage 数: {avg_relevant:.2f}")
    print("=" * 50 + "\n")


def main():
    # 数据路径
    data_dir = "data"
    
    # 输入文件
    collection_path = os.path.join(data_dir, "collection.tsv")
    queries_train_path = os.path.join(data_dir, "queries.train.tsv")
    queries_dev_path = os.path.join(data_dir, "queries.dev.tsv")
    qrels_train_path = os.path.join(data_dir, "qrels.train.tsv")
    qrels_dev_path = os.path.join(data_dir, "qrels.dev.tsv")
    
    # 输出文件
    corpus_output = os.path.join(data_dir, "marco-passages.jsonl")
    train_output = os.path.join(data_dir, "marco_train.parquet")
    dev_output = os.path.join(data_dir, "marco_dev.parquet")
    
    # 检查文件是否存在
    required_files = [collection_path, queries_train_path, qrels_train_path]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误: 找不到文件 {f}")
            print("请先运行 download_msmarco.ps1 下载数据，然后解压 tar.gz 文件")
            return
    
    # 1. 加载数据
    passages = load_passages(collection_path)
    queries_train = load_queries(queries_train_path)
    qrels_train = load_qrels(qrels_train_path)
    
    # 打印统计
    print_data_stats(qrels_train)
    
    # 2. 转换 corpus 格式（跳过已存在的文件）
    if os.path.exists(corpus_output):
        print(f"跳过: {corpus_output} 已存在 ({os.path.getsize(corpus_output) / 1e9:.2f} GB)")
    else:
        convert_to_search_r1_corpus(passages, corpus_output)
    
    # 3. 准备训练数据（跳过已存在的文件）
    if os.path.exists(train_output):
        print(f"跳过: {train_output} 已存在")
    else:
        prepare_training_data(queries_train, qrels_train, train_output)
    
    # 4. 准备开发集（如果有，跳过已存在的文件）
    if os.path.exists(queries_dev_path) and os.path.exists(qrels_dev_path):
        if os.path.exists(dev_output):
            print(f"跳过: {dev_output} 已存在")
        else:
            queries_dev = load_queries(queries_dev_path)
            qrels_dev = load_qrels(qrels_dev_path)
            print_data_stats(qrels_dev)
            prepare_training_data(queries_dev, qrels_dev, dev_output)
    
    print("\n" + "=" * 50)
    print("数据处理完成！")
    print("=" * 50)
    print(f"\n生成的文件:")
    print(f"  - {corpus_output} (用于构建 FAISS 索引)")
    print(f"  - {train_output} (训练数据)")
    print(f"  - {dev_output} (开发集数据)")
    print(f"\n下一步:")
    print(f"  1. 运行 build_faiss_index.py 构建 FAISS 索引")
    print(f"  2. 启动 retrieval_server.py")
    print(f"  3. 运行 o365_search_agent.py 开始训练")


if __name__ == "__main__":
    main()
