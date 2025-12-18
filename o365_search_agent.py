# Copyright (c) Microsoft. All rights reserved.

"""
O365 Search Agent 训练脚本

与 search_r1 的区别：
1. 输出格式: `<search>o365_search(query="xxx")</search>`
2. Reward: NDCG (基于 MS MARCO qrels)

Usage:
    python o365_search_agent.py
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import logging

import pandas as pd
import requests
from openai import OpenAI

from agentlightning import LLM, LitAgent, NamedResources, Rollout
from agentlightning.logging import configure_logger, setup

# 同时输出到控制台和文件，便于回溯每个 rollout 的 reward / query / docids
LOG_FILE_PATH = os.path.join("logs", "o365_rollouts.log")
setup(level=logging.INFO, files=LOG_FILE_PATH, console=True, color=False)
logger = configure_logger(name=__name__)

# O365 Search Agent 的系统提示
INSTRUCTION_FORMAT = """You are an O365 search assistant. Based on the user's question, invoke a search to find relevant documents.

You must first perform reasoning and analysis within <think></think>.

Then, you must perform the search using the format <search>o365_search(query="query keywords")</search>.

The search results will be returned in <information></information>.

Once an answer is found, provide the answer in <answer></answer>.

Important Notes:
- The query should be the key search term extracted from the user's question.
- You can rewrite a user's question into a more efficient search query, but you must ensure that the original meaning is not changed..
- Use concise and accurate keywords, removing irrelevant words.
- You can perform multiple searches to refine the results.

Example:
User Question: What are the symptoms of diabetes?

<think>User wants to know about the symptoms of diabetes, I need to search for relevant medical information</think>
<search>o365_search(query="diabetes symptoms")</search>

Question: """


class Document(TypedDict):
    docid: str
    contents: str


class RetrievalItem(TypedDict):
    document: Document
    score: float


def compute_ndcg(
    retrieved_docids: List[str],
    relevant_passages: Dict[str, int],
    k: int = 10
) -> float:
    """
    计算 NDCG@k

    Args:
        retrieved_docids: 检索返回的 docid 列表（按排序）
        relevant_passages: qrels 中的相关性标注 {docid: relevance}
        k: 截断位置

    Returns:
        NDCG@k 分数 (0.0 ~ 1.0)
    """
    if not relevant_passages:
        return 0.0

    # 计算 DCG@k
    dcg = 0.0
    for i, docid in enumerate(retrieved_docids[:k]):
        rel = relevant_passages.get(docid, 0)
        # DCG 公式: rel_i / log2(i + 2)
        dcg += rel / math.log2(i + 2)

    # 计算 IDCG@k (理想情况下的 DCG)
    ideal_rels = sorted(relevant_passages.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def extract_o365_search_query(response: str) -> Optional[str]:
    """
    从响应中提取 o365_search 的 query 参数

    支持格式:
    - <search>o365_search(query="xxx")</search>
    - <search>o365_search(query='xxx')</search>
    """
    # 匹配 o365_search(query="xxx") 或 o365_search(query='xxx')
    pattern = r'<search>\s*o365_search\s*\(\s*query\s*=\s*["\']([^"\']+)["\']\s*\)\s*</search>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # 兼容旧格式 <search>query</search>
    fallback_pattern = r'<search>([^<]+)</search>'
    fallback_match = re.search(fallback_pattern, response, re.DOTALL)
    if fallback_match:
        content = fallback_match.group(1).strip()
        # 如果不是 o365_search 格式，直接作为 query
        if 'o365_search' not in content:
            return content

    return None


def postprocess_response(response: str) -> str:
    """处理响应，在 search 或 answer 标签后截断"""
    if "</search>" in response:
        response = response.split("</search>")[0] + "</search>"
    elif "</answer>" in response:
        response = response.split("</answer>")[0] + "</answer>"
    return response


def extract_action(response: str) -> Tuple[Optional[str], str]:
    """从响应中提取动作类型和内容"""
    # 先检查是否是 answer
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    # 检查是否是 search
    search_query = extract_o365_search_query(response)
    if search_query:
        return "search", search_query

    return None, ""


def retrieve_doc(query: str, topk: int = 10, return_docids: bool = True) -> Tuple[str, List[str]]:
    """
    调用检索服务获取文档

    Args:
        query: 搜索查询
        topk: 返回文档数量
        return_docids: 是否返回 docid 列表

    Returns:
        (格式化的文档字符串, docid 列表)
    """
    payload: Dict[str, Any] = {"queries": [query], "topk": topk, "return_scores": True}
    response = requests.post("http://127.0.0.1:8000/retrieve", json=payload)
    response.raise_for_status()
    json_resp: Dict[str, Any] = cast(Dict[str, Any], response.json())
    retrieval_result: List[RetrievalItem] = cast(List[RetrievalItem], json_resp["result"][0])

    # 提取 docids 用于 NDCG 计算
    docids = [item["document"]["docid"] for item in retrieval_result]

    # 格式化文档字符串
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        docid = doc_item["document"]["docid"]
        content = doc_item["document"]["contents"]
        # 截断过长的内容
        if len(content) > 500:
            content = content[:500] + "..."
        format_reference += f"Doc {idx+1} (ID: {docid}): {content}\n"

    return format_reference, docids


def execute_response(
    response: str,
    relevant_passages: Dict[str, int],
    topk: int = 10,
    do_search: bool = True
) -> Tuple[str, float]:
    """
    执行响应中的动作

    Returns:
        (环境反馈字符串, NDCG 分数)
    """
    action, content = extract_action(response)

    if action == "answer":
        return "", 0.0
    elif action == "search":
        if do_search:
            search_result, retrieved_docids = retrieve_doc(content, topk=topk)
            # 计算 NDCG
            ndcg_score = compute_ndcg(retrieved_docids, relevant_passages, k=topk)
            # 记录本次搜索的检索 docid 和相关 docid，便于排查为什么得到当前 NDCG
            relevant_docids = list(relevant_passages.keys())
            logger.info(
                "[Search] query=%s ndcg=%.4f retrieved_topk=%s relevant=%s",
                content,
                ndcg_score,
                retrieved_docids,
                relevant_docids,
            )
            return f"\n\n<information>{search_result}</information>\n\n", ndcg_score
        else:
            return "", 0.0
    else:
        return (
            "\nMy previous action is invalid. To search, use: <search>o365_search(query=\"your query\")</search>. "
            "To answer, use: <answer>your answer</answer>. Let me try again.\n",
            0.0
        )


def call_llm(
    llm_client: OpenAI,
    model_name: str,
    content: str,
    temperature: float = 1.0,
    max_tokens: int = 500,
) -> str:
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


class O365SearchAgent(LitAgent[Dict[str, Any]]):
    """
    O365 搜索 Agent

    使用 NDCG 作为 reward，训练模型生成更好的搜索 query
    """

    def __init__(
        self,
        val_temperature: Optional[float] = 0.0,
        max_turns: int = 4,
        topk: int = 5,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.data_dir = os.environ.get("VERL_O365_DATA_DIR", "data")
        self.max_turns = max_turns
        self.topk = topk

    def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        prompt = INSTRUCTION_FORMAT + task["question"]
        # 解析 relevant_passages (JSON 字符串)
        relevant_passages_str = task.get("relevant_passages", "{}")
        if isinstance(relevant_passages_str, str):
            relevant_passages: Dict[str, int] = json.loads(relevant_passages_str)
        else:
            relevant_passages = relevant_passages_str

        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Question: {task['question']}")
        logger.info(f"[Rollout {rollout_id}] Relevant passages: {len(relevant_passages)} docs")
        logger.info(f"[Rollout {rollout_id}] Prompt(head): {prompt[:500]}")

        start_time = time.time()
        llm: LLM = cast(LLM, resources["main_llm"])
        client = OpenAI(
            base_url=llm.get_base_url(rollout_id, rollout.attempt.attempt_id),  # type: ignore[attr-defined]
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )

        if rollout.mode == "train":
            temperature = llm.sampling_parameters.get("temperature", 1.0)
        else:
            temperature = self.val_temperature if self.val_temperature is not None else 0.0

        turn_id = 0
        finished_flag = False
        rollout_content: str = ""
        best_ndcg: float = 0.0  # 记录最佳 NDCG 分数

        try:
            while turn_id < self.max_turns and not finished_flag:
                turn_id += 1
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                valid_turn_response = postprocess_response(turn_response)
                rollout_content += valid_turn_response

                turn_env_feedback, turn_ndcg = execute_response(
                    valid_turn_response, relevant_passages, topk=self.topk
                )

                # 更新最佳 NDCG
                if turn_ndcg > best_ndcg:
                    best_ndcg = turn_ndcg

                if len(turn_env_feedback) == 0:
                    finished_flag = True
                else:
                    rollout_content += turn_env_feedback

                logger.info(
                    f"TURN ID {turn_id} | RESP: {turn_response[:200]}... | "
                    f"NDCG: {turn_ndcg:.4f} | Best NDCG: {best_ndcg:.4f}"
                )

            # 如果没有正常结束，再生成一个回复
            if not finished_flag:
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                rollout_content += turn_response
                logger.info(f"LAST TURN GENERATE | RESP: {turn_response[:200]}...")

        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during rollout: {e}")
            return None

        end_time = time.time()

        # 使用最佳 NDCG 作为 reward
        reward_score = best_ndcg

        logger.info("[Rollout %s] Final Reward (Best NDCG): %.4f", rollout_id, reward_score)
        logger.info("[Rollout %s] Time taken: %.2f seconds", rollout_id, end_time - start_time)
        logger.info(
            "question: %s | best_ndcg: %.4f | turns: %d",
            task["question"][:50], reward_score, turn_id
        )

        return reward_score


def debug_o365_search_agent() -> None:
    """调试函数，用于本地测试 Agent"""
    data_path = os.path.join(os.environ.get("VERL_O365_DATA_DIR", "data"), "marco_train.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist. Run prepare_marco_data.py first.")

    df = pd.read_parquet(data_path).head(10)  # type: ignore[call-overload]
    print(f"Loaded {len(df)} samples for debugging")

    # 模拟测试（需要检索服务运行中）
    for _, row in df.iterrows():
        print(f"\nQuestion: {row['question']}")
        relevant_passages = json.loads(row['relevant_passages'])
        print(f"Relevant passages: {len(relevant_passages)} docs")

        # 模拟搜索
        try:
            query = row['question'][:50]  # 使用问题前50字符作为测试query
            _result_str, docids = retrieve_doc(query, topk=5)
            ndcg = compute_ndcg(docids, relevant_passages, k=5)
            print(f"Test query: '{query}'")
            print(f"Retrieved docids: {docids}")
            print(f"NDCG@5: {ndcg:.4f}")
        except Exception as e:
            print(f"Error (is retrieval server running?): {e}")
        break


if __name__ == "__main__":
    debug_o365_search_agent()
