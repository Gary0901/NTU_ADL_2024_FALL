#!/bin/bash

# 安裝 gdown（如果尚未安裝）
pip install gdown

# 創建必要的目錄
mkdir -p models_mq
mkdir -p models_qa
mkdir -p data
mkdir -p models

# 設置 Google Drive 文件夾 ID
FOLDER_ID="1JEC01vV4KJebLjBHEz7pZuAwZ1h0hIMH"

# 下載整個文件夾
gdown https://drive.google.com/drive/folders/${FOLDER_ID} -O ./temp_download --folder

# 移動文件到正確的位置
mv temp_download/data/*.json data/
mv temp_download/models/*.zip models/

# 解壓模型文件
unzip models/model_mq.zip -d models_mq
unzip models/model_qa.zip -d models_qa

# 清理臨時文件和資料夾
rm -rf temp_download
rm -rf models

echo "下載完成！"