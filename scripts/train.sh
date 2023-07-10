#!/bin/bash

declare -a models=(
        #"microsoft/deberta-v3-large"
        "MilaNLProc/deberta-v3-large-mlm-reddit-gab"
        #"Elron/deberta-v3-large-sentiment"
        #"ghatgetanuj/microsoft-deberta-v3-large_cls_SentEval-CR"
        "mrm8488/distilroberta-base-finetuned-suicide-depression"
        "rafalposwiata/deproberta-large-depression"
        #"rafalposwiata/roberta-large-depression"
        #"paulagarciaserrano/roberta-depression-detection"
        #"roberta-large"
)

DATA_CONFIG=${1}

for MODEL in "${models[@]}"; do
   python run_glue.py \
        --train_file data/train_${DATA_CONFIG}.csv \
        --validation_file data/dev_data.csv \
        --model_name_or_path ${MODEL} \
        --output_dir output/${DATA_CONFIG}/$(basename ${MODEL}) \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --save_total_limit 1 \
        --do_train \
        --do_eval
done
