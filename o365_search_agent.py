# Copyright (c) Microsoft. All rights reserved.
"""
O365 Search Agent 训练脚本

使用 MS MARCO 数据集，训练模型生成 o365_search(query="xxx") 格式的搜索指令
使用 NDCG 作为 reward
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from openai import OpenAI

from agentlightning import LLM, LitAgent, NamedResources, Trainer, reward, setup_logging

setup_logging()

# Prompt 模板
INSTRUCTION_FORMAT = """你是一个 O365 搜索助手。根据用户的问题，调用搜索找到相关文档。

你必须先在 <think> </think> 中进行推理分析。
然后使用 <search>o365_search(query="你的查询关键词")</search> 格式进行搜索。
搜索结果会返回在 <information> </information> 中。
找到答案后，在 <answer> </answer> 中给出答案。

重要提示：
- query 应该是从用户问题中提取的关键搜索词
- 使用简洁、准确的关键词，去掉无关的词语
- 可以多次搜索来完善结果

示例：
用户问题: What are the symptoms of diabetes?
<think>用户想了解糖尿病的症状，我需要搜索相关医学信息</think>
<search>o365_search(query="diabetes symptoms")</search>

问题："""


# ============================================================
# NDCG 计算
# ============================================================

def compute_dcg(relevances: List[float], k: int = 10) -> float:
    """计算 DCG@k"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        # DCG = sum(rel_i / log2(i+2))
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(retrieved_pids: List[str], relevance_map: Dict[str, float], k: int = 10) -> float:
    """
    计算 NDCG@k
    
    Args:
        retrieved_pids: 检索返回的 passage ids（按排名顺序）
        relevance_map: {pid: relevance_score} 来自 qrels
        k: 截断位置
    
    Returns:
        NDCG@k score (0.0 ~ 1.0)
    """
    if not retrieved_pids or not relevance_map:
        return 0.0
    
    # 获取检索结果的相关性分数
    relevances = [float(relevance_map.get(pid, 0)) for pid in retrieved_pids]
    dcg = compute_dcg(relevances, k)
    
    # 计算理想 DCG（所有相关文档按分数降序排列）
    ideal_relevances = sorted(relevance_map.values(), reverse=True)
    idcg = compute_dcg(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


@reward
async def eval_ndcg(retrieved_pids: List[str], relevance_map: Dict[str, float]) -> float:
    """计算 NDCG@10 作为 reward"""
    score = compute_ndcg(retrieved_pids, relevance_map, k=10)
    return score


# ============================================================
# 解析和执行
# ============================================================

def extract_search_query(response: str) -> Optional[str]:
    """
    从 response 中提取 o365_search 的 query 参数
    
    格式: <search>o365_search(query="xxx")</search>
    """
    # 匹配 o365_search(query="xxx") 或 o365_search(query='xxx')
    pattern = r'<search>o365_search\(query=["\'](.+?)["\']\)</search>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer(response: str) -> Optional[str]:
    """提取 <answer>xxx</answer>"""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def postprocess_response(response: str) -> str:
    """截断到第一个完整的 tag"""
    if "</search>" in response:
        return response.split("</search>")[0] + "</search>"
    elif "</answer>" in response:
        return response.split("</answer>")[0] + "</answer>"
    return response


def retrieve_documents(query: str, topk: int = 10) -> Tuple[List[str], str]:
    """
    调用检索服务
    
    Returns:
        (passage_ids, formatted_text_for_llm)
    """
    try:
        payload = {"queries": [query], "topk": topk, "return_scores": True}
        resp = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=30)
        resp.raise_for_status()
        
        results = resp.json()["result"][0]
        
        pids = []
        formatted = ""
        for idx, item in enumerate(results):
            # docid 可能在不同字段
            pid = item.get("docid") or item.get("document", {}).get("docid") or str(idx)
            content = item.get("document", {}).get("contents", "")
            
            pids.append(str(pid))
            # 截断内容避免太长
            content_preview = content[:300] + "..." if len(content) > 300 else content
            formatted += f"Doc {idx+1} (ID: {pid}): {content_preview}\n\n"
        
        return pids, formatted
    
    except Exception as e:
        print(f"Retrieval error: {e}")
        return [], f"搜索出错: {e}"


def call_llm(client: OpenAI, model: str, content: str, temperature: float = 1.0) -> str:
    """调用 LLM"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content or ""


# ============================================================
# Agent 定义
# ============================================================

class O365SearchAgent(LitAgent[Any]):
    """O365 搜索 Agent"""
    
    async def training_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
        temperature: float = 1.0,
    ) -> Any:
        """训练时的 rollout"""
        
        # 获取任务数据
        prompt = INSTRUCTION_FORMAT + task["question"]
        relevance_map: Dict[str, float] = task.get("relevant_passages", {})
        
        # 转换 relevance_map 的 key 为字符串
        relevance_map = {str(k): float(v) for k, v in relevance_map.items()}
        
        # 获取 LLM
        llm: LLM = cast(LLM, resources.get("main_llm"))
        client = OpenAI(
            base_url=llm.endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )

        turn_id = 0
        finished = False
        rollout_content = ""
        all_retrieved_pids: List[str] = []

        # 最多 4 轮搜索
        while turn_id < 4 and not finished:
            turn_id += 1
            
            # 生成 response
            response = call_llm(client, llm.model, prompt + rollout_content, temperature)
            response = postprocess_response(response)
            
            # 检查是否是 answer
            if extract_answer(response):
                finished = True
                rollout_content += response
                break
            
            # 检查是否是 search
            query = extract_search_query(response)
            if query:
                pids, doc_text = retrieve_documents(query, topk=10)
                all_retrieved_pids.extend(pids)
                feedback = f"\n\n<information>\n{doc_text}</information>\n\n"
                print(f"Turn {turn_id} | Query: '{query}' | Retrieved: {len(pids)} docs")
            else:
                feedback = (
                    "\n格式错误。请使用正确格式: "
                    '<search>o365_search(query="你的查询")</search>\n'
                )
                print(f"Turn {turn_id} | Invalid format")
            
            rollout_content += response + feedback

        # 如果还没结束，强制生成答案
        if not finished:
            response = call_llm(client, llm.model, prompt + rollout_content, temperature)
            rollout_content += response

        # 计算 NDCG reward
        # 只取前 10 个唯一的 pid
        unique_pids = list(dict.fromkeys(all_retrieved_pids))[:10]
        reward_score = await eval_ndcg(unique_pids, relevance_map)
        
        print(f"Question: {task['question'][:50]}...")
        print(f"Retrieved PIDs: {unique_pids[:5]}...")
        print(f"Relevant PIDs: {list(relevance_map.keys())[:5]}...")
        print(f"NDCG@10: {reward_score:.4f}")
        print("-" * 50)
        
        return reward_score

    async def validation_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
    ) -> Any:
        """验证时使用 temperature=0"""
        return await self.training_rollout_async(task, resources, rollout, temperature=0.0)


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    # 启动训练
    # n_workers: 并行 worker 数量
    Trainer(n_workers=128).fit(O365SearchAgent(), "http://localhost:9999/")
