MODEL_NAME="base-7b-noCoT"
DATASET="Controlled_Images_A"

torchrun \
    main.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --download  #--download #the first time needs download of dataset