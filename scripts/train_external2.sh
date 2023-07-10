#!/bin/bash

export WANDB_PROJECT="Depsign"

MODEL="output/external_roberta-large-mental-health-v1_dep_cds_backup"
DATASET="dep_cds"
MODEL_NAME="roberta-large-mental-health-v1"
CONFIG_DIR="config/external2/${MODEL_NAME}"

export WANDB_TAGS="${MODEL_NAME}","${DATASET}"

while read line; do
    echo ${line//MODEL/${MODEL}}
done < "${CONFIG_DIR}"/tune.yaml > "${CONFIG_DIR}"/tune.temp

datalawyer_tune tune huggingface_script \
"${CONFIG_DIR}"/tune.temp \
"${CONFIG_DIR}"/hparams.json \
--optuna-param-path "${CONFIG_DIR}"/config.json \
--serialization-dir output/external2_${MODEL_NAME}_${DATASET} \
--metrics "eval_f1 (macro)" \
--study-name ${WANDB_TAGS} \
--direction maximize \
--skip-if-exists \
--storage sqlite:///db.sqlite3




