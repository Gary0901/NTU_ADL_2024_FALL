#!/bin/bash

MODEL_DIR="./models/checkpoint-6513/checkpoint-6513"
INPUT_FILE="./data/public.jsonl"
OUTPUT_FILE="./output.jsonl"

python script/inference.py \
    --model_path "$MODEL_DIR" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --strategy "beam_search"

echo "Inference completed."