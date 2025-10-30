#!/usr/bin/env bash
TRAIN_DATA=""
TEST_DATA=""
VAL_DATA=""

MAX_LENGTH=128
BATCH_SIZE=32
LEARNING_RATE=5e-5
NUM_EPOCHS=3
MAX_SAMPLES=16000
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
