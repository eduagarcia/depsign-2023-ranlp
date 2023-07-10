#!/bin/bash

export WANDB_PROJECT="depsign_external"

MODEL="edu/models/roberta-large-mental-health-v1"
MODEL_NAME=$(basename ${MODEL})

CONFIG_DIR="config/external/${MODEL_NAME}"

declare -a datasets=(
    "dep_cds"
)

for DATASET in "${datasets[@]}"; do
    export WANDB_TAGS="${DATASET},${MODEL_NAME}"
    
    while read line; do
        line=${line//DATASET/${DATASET}}
        echo ${line//MODEL/${MODEL}}
    done < "${CONFIG_DIR}"/tune.yaml > "${CONFIG_DIR}"/tune.temp

    datalawyer_tune tune huggingface_script \
    "${CONFIG_DIR}"/tune.temp \
    "${CONFIG_DIR}"/hparams.json \
    --optuna-param-path "${CONFIG_DIR}"/config.json \
    --serialization-dir output/external_${MODEL_NAME}_${DATASET} \
    --metrics "eval_f1 (macro)" \
    --study-name ${WANDB_TAGS}_2\
    --direction maximize \
    --skip-if-exists \
    --storage sqlite:///db.sqlite3
done



