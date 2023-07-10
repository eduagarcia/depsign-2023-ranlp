#!/bin/bash

declare -a models=(
    #"MilaNLProc/deberta-v3-large-mlm-reddit-gab"
    "mrm8488/distilroberta-base-finetuned-suicide-depression"
    "rafalposwiata/deproberta-large-depression"
)

DATA_CONFIG="duplicates"

for MODEL in "${models[@]}"; do
    python multiclass_classification.py \
        --train_file data/train_${DATA_CONFIG}.csv \
        --validation_file data/dev_data.csv \
        --model_name_or_path ${MODEL} \
        --output_dir output/${DATA_CONFIG}-class_weights/$(basename ${MODEL}) \
        --batch_size 16
done
