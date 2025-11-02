MODEL_PATH="/path/to/local/Spatial-SSRL-7B"
INPUT_JSON="VSIBench/vsi_bench.json"
OUTPUT_DIR="CoT"
OUTPUT_JSON="Spatial-SSRL-7B.jsonl"
BATCH=1
MAX_NEW_TOKENS=4096


torchrun \
    --nproc-per-node=8 \
    vsi_infer.py \
    --model_path $MODEL_PATH \
    --input_json  $INPUT_JSON \
    --output_dir  $OUTPUT_DIR \
    --output_json $OUTPUT_JSON \
    --max_new_tokens $MAX_NEW_TOKENS \
    --cot True