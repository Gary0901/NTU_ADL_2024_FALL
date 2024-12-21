#!/bin/bash 
CONTEXT_FILE="${1:-./data/context.json}"
TEST_FILE="${2:-./data/test.json}"
OUTPUT_FILE="${3:-./prediction.csv}"

MQ_MODEL_DIR="./models_mq"
QA_MODEL_DIR="./models_qa"


python script/inference.py \
    --mq_model_dir "$MQ_MODEL_DIR" \
    --qa_model_dir "$QA_MODEL_DIR" \
    --test_file "$TEST_FILE" \
    --context_file "$CONTEXT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --pretrained_model "hfl/chinese-lert-base"

echo "Inference completed. Output save to $OUTPUT_FILE"