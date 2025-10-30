#!/bin/bash
# Set parameters
TRAIN_DATA="" #real data 6590

TEST_DATA=""



DATASET="semeval2010"  # 2010 2018 chemprot ddi semeval2010
LEARNING_RATE=1e-5
NUM_EPOCHS=3

SEED=42
MODEL="distilroberta-base"

python Downstream/relation_extraction.py \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --dataset $DATASET \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --model $MODEL \
  --seed $SEED 