#!/usr/bin/env bash


TRAIN_DATA=""
TEST_DATA=""
VAL_DATA=""

MAX_INPUT_LENGTH=256
MAX_TARGET_LENGTH=128
BATCH_SIZE=32
LEARNING_RATE=3e-5
NUM_EPOCHS=20
SEED=42
MODEL="t5-large"
MAX_SAMPLES=3000

python Downstream/t5_summarization.py \
  --train_data "$TRAIN_DATA" \
  --test_data "$TEST_DATA" \
  --val_data "$VAL_DATA" \
  --max_input_length $MAX_INPUT_LENGTH \
  --max_target_length $MAX_TARGET_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --max_samples $MAX_SAMPLES \
  --seed $SEED \
  --model "$MODEL" \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1
