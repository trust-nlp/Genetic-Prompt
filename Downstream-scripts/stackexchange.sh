#!/usr/bin/env bash


TRAIN_DATA=""
TEST_DATA=""
VAL_DATA=""

MAX_LENGTH=256
BATCH_SIZE=16
LEARNING_RATE=5e-5
NUM_EPOCHS=10
MAX_SAMPLES=27000
SEED=42
MODEL_NAME="roberta-base"

python Downstream/classification.py \
  --train_data "$TRAIN_DATA" \
  --test_data "$TEST_DATA" \
  --val_data "$VAL_DATA" \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --max_samples $MAX_SAMPLES \
  --seed $SEED \
  --model_name "$MODEL_NAME"
