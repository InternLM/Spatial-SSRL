MODEL_PATH="internlm/Spatial-SSRL-7B"

torchrun \
    --nproc-per-node=8 \
    main.py \
    --model_path $MODEL_PATH \
    --CoT \
    --verbose