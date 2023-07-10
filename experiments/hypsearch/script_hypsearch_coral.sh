#!/usr/bin/env bash

# Requires install of datalawyer_tune:
# https://bitbucket.org/avisourgente/datalawyer-tune

CONFIG_DIR=hypsearch_configs/coral-roberta-mental-health-v2
STUDY_NAME=coral-roberta-mental-health-v2
OUTPUT_DIR=output/${STUDY_NAME}
#BATCH_SIZE=1
#NUM_GRADIENT_ACCUMULATION_STEPS=16
export WANDB_PROJECT="Depsign"
export WANDB_TAGS="${STUDY_NAME},coral,head+tail"

# export num_gradient_accumulation_steps=${NUM_GRADIENT_ACCUMULATION_STEPS}
# export batch_size=${BATCH_SIZE}

datalawyer_tune tune huggingface_script \
  "${CONFIG_DIR}"/tune.yaml \
  "${CONFIG_DIR}"/hparams.json \
  --optuna-param-path "${CONFIG_DIR}"/config.json \
  --serialization-dir "${OUTPUT_DIR}" \
  --metrics "eval_f1 (macro)" \
  --study-name "${STUDY_NAME}" \
  --direction maximize \
  --skip-if-exists \
  --storage sqlite:///db.sqlite3
