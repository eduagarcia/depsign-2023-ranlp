#!/bin/bash

TASK_TYPE="regression"
CONFIG_DIR="config/${TASK_TYPE}"
TIMESTAMP=$(date +%s)


export WANDB_PROJECT="Depsign"
export WANDB_TAGS="regression"



datalawyer_tune tune huggingface_script \
"${CONFIG_DIR}"/tune.yaml \
"${CONFIG_DIR}"/hparams.json \
--optuna-param-path "${CONFIG_DIR}"/config.json \
--serialization-dir output/${TASK_TYPE} \
--metrics "eval_f1 (macro)" \
--study-name ${WANDB_TAGS}_${TIMESTAMP} \
--direction maximize \
--skip-if-exists \
--storage sqlite:///db.sqlite3




