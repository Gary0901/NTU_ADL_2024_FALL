#!/bin/bash

# 創建 adapter_checkpoint 資料夾
mkdir -p adapter_checkpoint

# 下載壓縮檔 (使用完整的 Google Drive uc URL)
echo "開始下載資料..."
gdown https://drive.google.com/uc?id=14AD_B35WzaD3EYoK-7pYTs2sC0sj2ksm -O model_result.zip

# 解壓縮到 adapter_checkpoint 資料夾
unzip model_result.zip -d adapter_checkpoint/

# 清理壓縮檔
rm model_result.zip