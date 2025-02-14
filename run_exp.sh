#!/bin/bash

POP_SIZE=30
MUTATE_RATE=0.02
CROSS_RATE=0.5
CROSS_STYLE="cols"
N_TRIALS=5000
INPUT_NODES=0
OUTPUT_NODES=0
ORDER=10
TASK="--task narma"
# METRIC="--metric KR"
EXPERIMENT_ID=0

NUM_JOBS=150

seq 1 $NUM_JOBS | parallel -j $NUM_JOBS python3 ../../main.py \
  --pop_size $POP_SIZE \
  --mutate_rate $MUTATE_RATE \
  --cross_rate $CROSS_RATE \
  --cross_style $CROSS_STYLE \
  --n_trials $N_TRIALS \
  --input_nodes $INPUT_NODES \
  --output_nodes $OUTPUT_NODES \
  --order $ORDER \
  $TASK \
  $METRIC \
  --output_file '{= $_ = ($ENV{EXPERIMENT_ID} + $_) . ".parquet" =}' \
  --exp_id '{= $_ = $ENV{EXPERIMENT_ID} + $_ =}'