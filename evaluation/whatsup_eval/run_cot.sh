MODEL_NAME="SpatialSSRL-7b"
DATASET="Controlled_Images_A"

torchrun \
    main.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --CoT \
    --download  #--download #the first time needs download of dataset