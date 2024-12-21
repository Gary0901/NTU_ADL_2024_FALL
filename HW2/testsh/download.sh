#!/bin/bash

# 安裝 gdown（如果尚未安裝）
# pip install gdown

# 創建必要的目錄
mkdir -p models
mkdir -p data

# 設置 Google Drive 文件夾 ID
FOLDER_ID="1PtxCJyg8ZrD15I9I6FGYLG7tvUqSKDIM"

# 創建臨時下載目錄
mkdir -p temp_download

# 下載整個文件夾
gdown https://drive.google.com/drive/folders/${FOLDER_ID} -O ./temp_download --folder

# 移動文件到正確的位置
mv temp_download/data/public.jsonl data/
mv temp_download/model/checkpoint-6513.zip models/

# 解壓模型文件
unzip models/checkpoint-6513.zip -d models/checkpoint-6513

# 清理臨時文件和資料夾
rm -rf temp_download

echo "下載完成！"