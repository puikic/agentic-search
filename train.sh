#!/bin/bash
# O365 Search Agent 训练启动脚本

# 配置
MODEL_PATH="meta-llama/Llama-3.2-3B"  # 或你自己的模型路径
DATA_DIR="data"

python -m agentlightning.runner.verl_runner \
    verl.algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/marco_train.parquet \
    data.val_files=${DATA_DIR}/marco_dev.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='o365-search-agent' \
    trainer.experiment_name='grpo-marco-ndcg' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.total_epochs=3
