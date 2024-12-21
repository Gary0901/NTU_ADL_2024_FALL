#!/bin/bash 

#參數說明
#BASE_MODEL_PATH=$1
#ADAPTER_PATH=$2
#INPUT_FILE=$3
#OUTPUT_FILE=$4

BASE_MODEL_PATH="zake7749/gemma-2-2b-it-chinese-kyara-dpo"
ADAPTER_PATH="./adapter_checkpoint"
INPUT_FILE="./data/private_test.json"
OUTPUT_FILE="./data/prediction.json"

# 檢查輸出目錄是否存在，若不存在則創建
OUTPUT_DIR=$(dirname "${OUTPUT_FILE}")
if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

# 執行預測
echo "開始執行推理..."
python script/inference.py \
    --base_model ${BASE_MODEL_PATH} \
    --adapter_model ${ADAPTER_PATH} \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE} 

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo "推理完成！輸出檔案: ${OUTPUT_FILE}"
else
    echo "推理過程發生錯誤"
    exit 1
fi
    