#!/bin/bash

source .venv/bin/activate

if [ "$1" == "--batch" ]; then
    python batch_predict.py \
        --dataset "${2:-./dataset}" \
        --output "${3:-submission.json}" \
        --config config/config.yaml \
        ${@:4}
else
    python predict.py \
        --video "$1" \
        --ref-images "$2" "$3" "$4" \
        --output "${5:-predictions.json}" \
        --config config/config.yaml \
        ${@:6}
fi
