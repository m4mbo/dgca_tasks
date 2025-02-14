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
# VERBOSE="--verbose" 
OUTPUT_FILE="fitness.parquet"
EXPERIMENT_ID=0  

# number of parallel jobs
NUM_JOBS=150

generate_command() {
  EXP_ID=$((EXPERIMENT_ID + $1)) # increment id
  echo "python3 main.py --pop_size $POP_SIZE --mutate_rate $MUTATE_RATE --cross_rate $CROSS_RATE --cross_style $CROSS_STYLE --n_trials $N_TRIALS --input_nodes $INPUT_NODES --output_nodes $OUTPUT_NODES --order $ORDER $TASK $METRIC $VERBOSE --output_file $OUTPUT_FILE --exp_id $EXP_ID"
}

export -f generate_command
export POP_SIZE MUTATE_RATE CROSS_RATE CROSS_STYLE N_TRIALS INPUT_NODES OUTPUT_NODES ORDER TASK METRIC VERBOSE OUTPUT_FILE EXPERIMENT_ID

# run the command in parallel
seq 1 $NUM_JOBS | parallel -j $NUM_JOBS generate_command {}