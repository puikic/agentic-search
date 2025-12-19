# Copyright (c) Microsoft. All rights reserved.

"""
O365 Search Agent 训练脚本

使用 VERL 框架进行 GRPO 训练，优化搜索 query 生成能力。
Reward 使用 NDCG@k 评估检索质量。

Usage:
    python train_o365_search_agent.py [fast|qwen|llama]

Examples:
    # 快速测试（CI）
    python train_o365_search_agent.py fast

    # 使用 Qwen2.5-7B 训练
    python train_o365_search_agent.py qwen

    # 使用 LLaMA-3.2-3B 训练
    python train_o365_search_agent.py llama
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from o365_search_agent import O365SearchAgent

import agentlightning as agl

# 基础训练配置
RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/marco_train.parquet",
        "val_files": "data/marco_dev.parquet",
        "train_batch_size": 192,
        "max_prompt_length": 2048,
        "max_response_length": 1024,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 3,  # 每个 prompt 生成 3 个样本，兼顾吞吐与多样性
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.9,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 128,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6, "lr_warmup_steps_ratio": 0.95},
            "use_kl_loss": True,
            "kl_loss_type": "low_var_kl",
            "kl_loss_coef": 0.001,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 4,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-7B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 8,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "O365SearchAgent",
        "experiment_name": "o365_search_ndcg",
        "nnodes": 1,
        "test_freq": 10,
        "save_freq": 10,
        "total_epochs": 10,
        "total_training_steps": 2000,
        "default_local_dir": "checkpoints/o365_search_checkpoints/",
    },
}


def config_train_fast() -> Dict[str, Any]:
    """快速训练配置，用于 CI 测试"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"o365_search_fast_{timestamp}"
    PROJECT_NAME = "O365SearchAgentCI"

    # 写入 GitHub Actions 输出
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-0.5B-Instruct"
    config["data"]["train_batch_size"] = 32
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """Qwen2.5-7B-Instruct 训练配置"""
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-7B-Instruct"
    config["trainer"]["experiment_name"] = "o365_search_qwen2.5_7b"
    return config


def config_train_qwen_3b() -> Dict[str, Any]:
    """Qwen2.5-3B-Instruct 训练配置（较小模型，适合资源有限场景）"""
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-3B-Instruct"
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["trainer"]["experiment_name"] = "o365_search_qwen2.5_3b"
    return config


def config_train_llama() -> Dict[str, Any]:
    """LLaMA-3.2-3B-Instruct 训练配置

    需要设置 HF_TOKEN 环境变量
    """
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] = "llama3_json"
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] = "llama3_json"
    config["actor_rollout_ref"]["model"]["path"] = "meta-llama/Llama-3.2-3B-Instruct"
    config["trainer"]["experiment_name"] = "o365_search_llama3.2_3b"
    return config


def train(config: Dict[str, Any]) -> None:
    """启动训练"""
    # 创建 Agent（使用 NDCG@5 reward，适合平均 1.2 个相关文档的数据集）
    agent = O365SearchAgent(
        val_temperature=0.0,
        max_turns=4,
        topk=5,
    )

    # 创建 VERL 算法
    algorithm = agl.VERL(config)

    # 创建 Trainer
    trainer = agl.Trainer(n_runners=16, algorithm=algorithm)

    # 加载数据
    train_data_path = config["data"]["train_files"]
    val_data_path = config["data"]["val_files"]

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Training data not found: {train_data_path}\n"
            "Please run prepare_marco_data.py first."
        )

    train_data = pd.read_parquet(train_data_path).to_dict(orient="records")  # type: ignore[call-overload]
    print(f"Loaded {len(train_data)} training samples")

    if os.path.exists(val_data_path):
        val_data = pd.read_parquet(val_data_path).to_dict(orient="records")  # type: ignore[call-overload]
        print(f"Loaded {len(val_data)} validation samples")
    else:
        # 如果没有验证集，使用训练集的前 1000 条
        val_data = train_data[:1000]
        train_data = train_data[1000:]  # ← 训练集要去掉这 1000 条
        print(f"No validation data found, split from training set: "
        f"{len(train_data)} train, {len(val_data)} val")

    # 开始训练
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore[arg-type]


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Train O365 Search Agent with NDCG reward"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen", "qwen3b", "llama"],
        nargs="?",
        default="qwen",
        help=(
            "Training configuration:\n"
            "  fast   - Quick test for CI\n"
            "  qwen   - Qwen2.5-7B-Instruct (default)\n"
            "  qwen3b - Qwen2.5-3B-Instruct\n"
            "  llama  - LLaMA-3.2-3B-Instruct"
        ),
    )

    args = parser.parse_args()

    config_functions = {
        "fast": config_train_fast,
        "qwen": config_train_qwen,
        "qwen3b": config_train_qwen_3b,
        "llama": config_train_llama,
    }

    config = config_functions[args.config]()

    print(f"\n{'=' * 50}")
    print(f"Starting O365 Search Agent training")
    print(f"Configuration: {args.config}")
    print(f"Model: {config['actor_rollout_ref']['model']['path']}")
    print(f"{'=' * 50}\n")

    train(config)


if __name__ == "__main__":
    main()
