#!/bin/bash
# Set parameters
TRAIN_DATA="" #real data 922


TEST_DATA=""
VAL_DATA=""  
DATASET="conll04"  # 2010 2018 chemprot ddi
LEARNING_RATE=1e-5
NUM_EPOCHS=3




python Downstream/relation_extraction.py \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --dataset $DATASET \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --seed $SEED \
  --model $MODEL 

