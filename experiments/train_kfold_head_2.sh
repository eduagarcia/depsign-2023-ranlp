#!/usr/bin/env bash

# Requires install of datalawyer_tune:
# https://bitbucket.org/avisourgente/datalawyer-tune

HYPSEARCH_DIR=hypsearch
CONFIG_DIR=${HYPSEARCH_DIR}/hypsearch_configs
export WANDB_PROJECT="kfold"
  
STUDY_NAME=oll-roberta-mental-health-v2-fixloss
OUTPUT_DIR=kfold/output/${STUDY_NAME}
export WANDB_TAGS=${STUDY_NAME},head+tail,oll

datalawyer_tune train huggingface_script \
  "${CONFIG_DIR}/${STUDY_NAME}/tune.yaml" \
  --overrides "{\"script_path\": \"${HYPSEARCH_DIR}/run_glue.py\", \"train_file\": \"/raid/juliana/depsign/edu/kfold/{0}/train_initial.csv\", \"validation_file\": \"/raid/juliana/depsign/edu/kfold/{0}/dev_data.csv\", \"test_file\": \"/raid/juliana/depsign/edu/kfold/{0}/dev_data.csv\", \"dropout_task\": 0.4, \"learning_rate\": 6e-6, \"oll_alpha\": 1.0}" \
  --aggregate "{\"fold\": [0, 1, 2, 3]}" \
  --serialization-dir "${OUTPUT_DIR}"