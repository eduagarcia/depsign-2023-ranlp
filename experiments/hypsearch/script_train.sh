#!/usr/bin/env bash

# Requires install of datalawyer_tune:
# https://bitbucket.org/avisourgente/datalawyer-tune

CONFIG_DIR=hypsearch_configs/roberta-mental-health-v4-headtail
STUDY_NAME=roberta-mental-health-v4-headtail_test
OUTPUT_DIR=output_train/${STUDY_NAME}/train_0
#BATCH_SIZE=1
#NUM_GRADIENT_ACCUMULATION_STEPS=16
export WANDB_PROJECT="Depsign"
export WANDB_TAGS=${STUDY_NAME},head+tail,test

# export num_gradient_accumulation_steps=${NUM_GRADIENT_ACCUMULATION_STEPS}
# export batch_size=${BATCH_SIZE}

datalawyer_tune train huggingface_script \
  "${CONFIG_DIR}"/tune.yaml \
  --serialization-dir "${OUTPUT_DIR}"
