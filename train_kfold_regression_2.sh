#!/usr/bin/env bash

# Requires install of datalawyer_tune:
# https://bitbucket.org/avisourgente/datalawyer-tune

HYPSEARCH_DIR=hypsearch
CONFIG_DIR=${HYPSEARCH_DIR}/hypsearch_configs
export WANDB_PROJECT="kfold"

STUDY_NAME=regression
OUTPUT_DIR=kfold/output/regression
export WANDB_TAGS=${STUDY_NAME}

datalawyer_tune train huggingface_script \
  "${CONFIG_DIR}/${STUDY_NAME}/tune.yaml" \
  --overrides "{\"script_path\": \"${HYPSEARCH_DIR}/run_glue.py\", \"train_file\": \"/raid/juliana/depsign/edu/kfold/{0}/train_initial.csv\", \"validation_file\": \"/raid/juliana/depsign/edu/kfold/{0}/dev_data.csv\", \"test_file\": \"/raid/juliana/depsign/edu/kfold/{0}/dev_data.csv\"}" \
  --aggregate "{\"fold\": [0, 1, 2, 3]}" \
  --serialization-dir "${OUTPUT_DIR}"
