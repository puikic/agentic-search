# O365 Search Agent 训练示例

本示例展示如何使用 Agent-Lightning 训练一个 O365 搜索 Agent，学习从用户自然语言查询中提取搜索关键词。

## 与 search_r1 的区别

| 项目 | search_r1 | O365 Search |
|------|-----------|-------------|
| 输出格式 | `<search>query</search>` | `<search>o365_search(query="xxx")</search>` |
| 数据集 | Wikipedia + NQ/HotpotQA | MS MARCO Passage Ranking |
| Reward | Exact Match | **NDCG** |

## 文件结构

```
o365_search/
├── download_msmarco.ps1       # 数据下载脚本 (Windows)
├── download_msmarco.sh        # 数据下载脚本 (Linux/Mac)
├── prepare_marco_data.py      # 数据预处理
├── build_faiss_index.py       # 构建 FAISS 索引
├── retrieval_server.py        # 检索服务
├── o365_search_agent.py       # Agent 定义 (NDCG reward)
├── train_o365_search_agent.py # RL 训练脚本
├── train.sh                   # 训练启动脚本
└── data/                      # 数据目录 (下载后生成)
    ├── collection.tsv         # 8.8M passages
    ├── queries.train.tsv      # 训练查询
    ├── qrels.train.tsv        # 相关性标注 (NDCG 关键!)
    ├── marco-passages.jsonl   # 转换后的 corpus
    ├── marco_e5.index         # FAISS 索引
    ├── marco_train.parquet    # 训练数据 	502,939 条
    └── marco_dev.parquet      # 测试数据 	55,578 条
```

## 快速开始

### Step 1: 下载数据

**Windows (PowerShell):**
```powershell
.\download_msmarco.ps1
```

**Linux/Mac:**
```bash
bash download_msmarco.sh
```

下载的文件说明：
- `collection.tar.gz` (~1GB): 8,841,823 passages
- `queries.tar.gz` (~42MB): 训练/开发集查询
- `qrels.train.tsv` (~10MB): **相关性标注，用于计算 NDCG**

### Step 2: 解压数据

**Windows (使用 Git Bash 或 7-Zip):**
```bash
cd data
tar -xzf collection.tar.gz
tar -xzf queries.tar.gz
```

### Step 3: 处理数据

```bash
python prepare_marco_data.py
```

这会生成：
- `marco-passages.jsonl`: 用于构建 FAISS 索引
- `marco_train.parquet`: 训练数据（包含 `relevant_passages` 用于 NDCG）

### Step 4: 构建 FAISS 索引

```bash
python build_faiss_index.py
```

⚠️ 注意：构建 8.8M passages 的索引需要较长时间和大内存。
可以先用 `--max_passages 100000` 参数测试。

### Step 5: 启动检索服务

```bash
python retrieval_server.py \
    --index_path data/marco_e5.index \
    --corpus_path data/marco-passages.jsonl \
    --topk 5 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2
```

### Step 6: 开始训练

```bash
# 先启动 Ray
bash ../../scripts/restart_ray.sh

# 方法 1: 使用 Python 脚本训练（推荐）
python train_o365_search_agent.py qwen    # Qwen2.5-7B
python train_o365_search_agent.py qwen3b  # Qwen2.5-3B (资源有限)
python train_o365_search_agent.py llama   # LLaMA-3.2-3B
python train_o365_search_agent.py fast    # 快速测试

# 方法 2: 使用 shell 脚本
bash train.sh
```

## NDCG 计算原理

MS MARCO 提供了 `qrels.train.tsv`，格式为：

```
qid    0    pid    relevance
1048585    0    7187158    1
```

这表示对于 query `1048585`，passage `7187158` 的相关性分数为 `1`。

NDCG 公式：

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

训练时，模型生成 `<search>o365_search(query="xxx")</search>`，
检索服务返回 top-k passages，根据 qrels 计算 NDCG 作为 reward。

## 模型输出格式

```xml
<think>用户想找Q3销售报告的PPT</think>
<search>o365_search(query="Q3 sales report presentation")</search>

<information>
Doc 1 (ID: 7187158): The Q3 sales report shows...
Doc 2 (ID: 7187160): Quarterly sales presentation...
</information>

<answer>根据搜索结果，Q3 sales report 相关的文档有...</answer>
```

## 参考资源

- [MS MARCO 官网](https://microsoft.github.io/msmarco/)
- [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking)
- [Search-R1 原项目](https://github.com/PeterGriffinJin/Search-R1)
