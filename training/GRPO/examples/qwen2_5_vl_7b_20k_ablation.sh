#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
MODEL_PATH=/mnt/shared-storage-user/mllm/shared/liuyuhong/LLaMA-Factory_2_5/output/qwen2_5vl_lora_0919_step75  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.train_files=/mnt/shared-storage-user/mllm/shared/liuyuhong/dataset/Shuffle_image/20k_ablation@train \
    data.val_files=/mnt/shared-storage-user/mllm/shared/liuyuhong/dataset/Shuffle_image/four-types-80k-0921@test \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_7b_20k_ablation \
    trainer.n_gpus_per_node=8 \
    worker.reward.reward_function=./examples/reward_function/math.py:img_score \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=5 \
    trainer.save_limit=3 \
    trainer.total_epochs=10 \
    trainer.val_freq=5 \
    worker.rollout.limit_images=5 \
    data.rollout_batch_size=256
