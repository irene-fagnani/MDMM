#!/bin/bash
# Usage: ./run_train.sh DATASET_NAME [MAX_IT] [--use_cpu] [--use_adain] [--double_layer] [--two_time_scale_rule RULE]

# Arguments
DATASET_NAME=$1
MAX_IT=$2
USE_CPU=$3          # pass exactly --use_cpu if you want it
USE_ADAIN=$4        # pass exactly --use_adain if you want it
DOUBLE_LAYER=$5     # pass exactly --double_layer if you want it
TWO_TS_RULE=$6      # pass one of: double_gen_enc, half_discr, none

# Set dataset-specific parameters
case "$DATASET_NAME" in
    summer2winter_yosemite)
        DATAROOT="../datasets/summer2winter_yosemite"
        ;;
    mini)
        DATAROOT="../datasets/mini"
        ;;
    orange2apple)
        DATAROOT="../datasets/orange2apple"
        ;;
    *)
        echo "Unknown dataset: $DATASET_NAME"
        exit 1
        ;;
esac

NAME=$DATASET_NAME
NUM_DOMAINS=2
DISPLAY_DIR="display_${DATASET_NAME}"
RESULT_DIR="results_${DATASET_NAME}"

# Build base Python command
CMD="python ../src/train.py --dataroot $DATAROOT --name $NAME --num_domains $NUM_DOMAINS --display_dir $DISPLAY_DIR --result_dir $RESULT_DIR"

# Add optional flags if provided
if [ "$USE_CPU" == "--use_cpu" ]; then
    CMD="$CMD --use_cpu"
fi

if [ "$USE_ADAIN" == "--use_adain" ]; then
    CMD="$CMD --use_adain"
fi

if [ "$DOUBLE_LAYER" == "--double_layer" ]; then
    CMD="$CMD --double_layer_ReLUINSConvTranspose"
fi

if [ -n "$TWO_TS_RULE" ]; then
    CMD="$CMD --two_time_scale_update_rule $TWO_TS_RULE"
fi

if [ -n "$MAX_IT" ]; then
    CMD="$CMD --max_it $MAX_IT"
fi

# Run the command
echo "Running command: $CMD"
eval $CMD

# Example usage: 

# chmod +x run_train.sh

# ./run_train.sh summer2winter_yosemite 5000
# Train summer2winter_yosemite with max 5000 iterations and CPU
# ./run_train.sh summer2winter_yosemite 5000 --use_cpu

# Train mini with AdaIN, double conv transpose, and two-time scale rule
# ./run_train.sh mini 10000 --use_adain --double_layer double_gen_enc

# Train orange2apple with default options
# ./run_train.sh orange2apple
