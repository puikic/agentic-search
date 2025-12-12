# Copyright (c) Microsoft. All rights reserved.
"""
构建 MS MARCO 的 FAISS 索引

使用 E5 模型将 passages 编码为向量，并构建 FAISS 索引
支持多 GPU 并行编码
"""

import argparse
import json
import os
from typing import List

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class E5Encoder:
    """E5 模型编码器，支持多 GPU"""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2", use_fp16: bool = True):
        print(f"Loading E5 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        self.num_gpus = torch.cuda.device_count()
        
        if self.num_gpus > 0:
            if use_fp16:
                self.model = self.model.half()
            
            if self.num_gpus > 1:
                # 多 GPU: 使用 DataParallel
                print(f"Using {self.num_gpus} GPUs with DataParallel")
                self.model = nn.DataParallel(self.model)
            else:
                print("Using 1 GPU with FP16")
            
            self.model = self.model.cuda()
        else:
            print("Using CPU (this will be slow!)")
    
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        """
        编码文本为向量
        
        E5 模型需要添加前缀:
        - query: "query: xxx"
        - passage: "passage: xxx"
        """
        all_embeddings = []
        
        # 添加前缀
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            hidden_state = outputs.last_hidden_state
            hidden_state = hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
        
        return np.vstack(all_embeddings)


def load_corpus(corpus_path: str, max_passages: int = None) -> tuple:
    """加载语料库"""
    print(f"Loading corpus from {corpus_path}")
    
    docids = []
    texts = []
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading corpus")):
            if max_passages and i >= max_passages:
                break
            
            doc = json.loads(line)
            docids.append(doc["docid"])
            # 截断到前 512 字符（E5 最大长度）
            texts.append(doc["contents"][:512])
    
    print(f"Loaded {len(docids)} passages")
    return docids, texts


def build_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """构建 FAISS 索引"""
    dim = embeddings.shape[1]
    print(f"Building FAISS index with {embeddings.shape[0]} vectors of dim {dim}")
    
    # 使用内积（因为已经 L2 normalize 了，内积等于余弦相似度）
    index = faiss.IndexFlatIP(dim)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Using GPU for FAISS")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors")
    
    return index


def save_index_and_mapping(index: faiss.Index, docids: List[str], output_dir: str):
    """保存索引和 docid 映射"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果是 GPU 索引，先转回 CPU
    try:
        # 尝试转换 GPU 索引到 CPU
        index = faiss.index_gpu_to_cpu(index)
        print("Converted GPU index to CPU")
    except Exception:
        # 已经是 CPU 索引，不需要转换
        pass
    
    # 保存 FAISS 索引
    index_path = os.path.join(output_dir, "marco_e5.index")
    faiss.write_index(index, index_path)
    print(f"Saved index to {index_path}")
    
    # 保存 docid 映射（索引位置 -> docid）
    mapping_path = os.path.join(output_dir, "docid_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(docids, f)
    print(f"Saved docid mapping to {mapping_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="data/marco-passages.jsonl")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="每个 GPU 的 batch size，8 GPU 时总 batch = 512*8=4096")
    parser.add_argument("--max_passages", type=int, default=None,
                        help="最大处理 passage 数，用于测试")
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument("--save_embeddings", action="store_true", default=True,
                        help="保存 embeddings 到文件，方便断点续传")
    parser.add_argument("--load_embeddings", type=str, default=None,
                        help="从文件加载 embeddings，跳过编码步骤")
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.corpus_path):
        print(f"错误: 找不到 {args.corpus_path}")
        print("请先运行 prepare_marco_data.py")
        return
    
    # 1. 加载语料库
    docids, texts = load_corpus(args.corpus_path, args.max_passages)
    
    # 2. 编码或加载 embeddings
    embeddings_path = os.path.join(args.output_dir, "embeddings.npy")
    docids_path = os.path.join(args.output_dir, "docids.json")
    
    if args.load_embeddings and os.path.exists(args.load_embeddings):
        # 从文件加载 embeddings（断点续传）
        print(f"Loading embeddings from {args.load_embeddings}...")
        embeddings = np.load(args.load_embeddings)
        print(f"Loaded embeddings shape: {embeddings.shape}")
    else:
        # 初始化编码器
        encoder = E5Encoder(args.model_name, args.use_fp16)
        
        # 编码所有 passages
        print("\nEncoding passages...")
        embeddings = encoder.encode(texts, batch_size=args.batch_size, is_query=False)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # 保存 embeddings（方便下次直接加载）
        if args.save_embeddings:
            print(f"Saving embeddings to {embeddings_path}...")
            np.save(embeddings_path, embeddings)
            with open(docids_path, "w") as f:
                json.dump(docids, f)
            print("Embeddings saved!")
    
    # 4. 构建索引 (用 CPU，避免 GPU 索引保存问题)
    index = build_index(embeddings, use_gpu=False)
    
    # 5. 保存
    save_index_and_mapping(index, docids, args.output_dir)
    
    print("\n" + "=" * 50)
    print("索引构建完成!")
    print("=" * 50)
    print(f"\n下一步: 启动检索服务")
    print(f"python retrieval_server.py \\")
    print(f"    --index_path {args.output_dir}/marco_e5.index \\")
    print(f"    --corpus_path {args.corpus_path} \\")
    print(f"    --retriever_model {args.model_name}")


if __name__ == "__main__":
    main()
