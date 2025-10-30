#!/usr/bin/env bash
TRAIN_DATA=""
TEST_DATA=""
DATASET="chemprot"
LEARNING_RATE=1e-5
NUM_EPOCHS=3
BATCH_SIZE=16
SEED=42
MODEL="roberta-base"

python Downstream/relation_extraction.py \
  --train_data "$TRAIN_DATA" \
  --test_data "$TEST_DATA" \
  --dataset "$DATASET" \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --seed $SEED \
  --model "$MODEL"
