#!/usr/bin/env bash

# Requires install of datalawyer_tune:
# https://bitbucket.org/avisourgente/datalawyer-tune

HYPSEARCH_DIR=hypsearch
CONFIG_DIR=${HYPSEARCH_DIR}/hypsearch_configs
export WANDB_PROJECT="kfold"

STUDY_NAME=roberta-large-v3-maxlen
OUTPUT_DIR=kfold/output/${STUDY_NAME}
export WANDB_TAGS=${STUDY_NAME}

#export batch_size=8
export num_gradient_accumulation_steps=2

datalawyer_tune retrain huggingface_script \
  "${CONFIG_DIR}/${STUDY_NAME}/tune.yaml" \
  --overrides "{\"script_path\": \"${HYPSEARCH_DIR}/run_glue.py\", \"train_file\": \"/raid/juliana/depsign/edu/kfold/{0}/train_initial.csv\", \"validation_file\": \"/raid/juliana/depsign/edu/kfold/{0}/dev_data.csv\", \"test_file\": \"/raid/juliana/depsign/edu/kfold/{0}/dev_data.csv\"}" \
  --aggregate "{\"fold\": [0, 1, 2, 3]}" \
  --serialization-dir "${OUTPUT_DIR}" \
  --study-name "${STUDY_NAME}" \
  --storage sqlite:///hypsearch/db.sqlite3
