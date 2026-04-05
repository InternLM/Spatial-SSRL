#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
#export WANDB_MODE=offline
MODEL_PATH=/path/to/Spatial-SSRL/training/SFT-coldstart/output/qwen2_5vl_7b_lora  # replace it with your local file path
TRAIN_FILE=/data/parquets@train
VAL_FILE=/data/parquets@test
IMG_DIR=/

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.image_dir=${IMG_DIR} \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_7b \
    trainer.n_gpus_per_node=8 \
    worker.reward.reward_function=./examples/reward_function/math.py:img_score \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=5 \
    trainer.save_limit=3 \
    trainer.total_epochs=10 \
    trainer.val_freq=5 \
    worker.rollout.limit_images=5 \
    data.rollout_batch_size=256
