#!/bin/bash


TRAIN_DATA="" #real data 1482

TEST_DATA=""
DATASET="ddi"  # ：2010 2018 chemprot ddi
LEARNING_RATE=1e-5
NUM_EPOCHS=3

SEED=2024
MODEL="roberta-base"
python Downstream/relation_extraction.py \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --dataset $DATASET \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --max_samples 3000 \
    --seed $SEED\
    --model $MODEL
