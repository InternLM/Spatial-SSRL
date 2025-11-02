MODEL_PATH="/path/to/local/models--Qwen--Qwen2.5-VL-7B-Instruct"

torchrun \
    --nproc-per-node=8 \
    main.py \
    --model_path $MODEL_PATH \
    --verbose # Set --verbose to print the responses during inference