# ADL HW1 - Chinese Question Answering

This repository contains the implementation of Chinese Question Answering system, including both paragraph selection and span selection tasks.

## Homework description
https://docs.google.com/presentation/d/1V5KE-AOTiVXWZjMVnr3GXyy_EjEMHeRKfeyudyeevl8/edit#slide=id.p

## File Structure
```
.
├── model/              # Model implementation
├── script/            # Training and inference scripts
├── download.sh        # Script to download required files
├── run.sh            # Script for inference
├── requirements.txt   # Package dependencies
└── README.md         # This file
```

## Environment Setup
```bash
# Install required packages
pip install -r requirements.txt
```

## Download Necessary Files
```bash
bash download.sh
```

## Training
1. Paragraph Selection Task
Fine-tune the LERT model for paragraph selection:
```
python3 model_mq.py \
    --model_name_or_path "hfl/chinese-lert-base" \
    --context_file "data/context.json" \
    --train_file "data/train.json" \
    --validation_file "data/valid.json" \
    --output_dir "output/lert_mq" \
    --max_seq_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2
```
The model will be saved in output/lert_mq directory.

2. Span Selection Task
Fine-tune the LERT model for span selection:
```
python3 model_qa.py \
    --model_name_or_path "hfl/chinese-lert-base" \
    --context_file "data/context.json" \
    --train_file "data/train.json" \
    --validation_file "data/valid.json" \
    --output_dir "output/lert_qa" \
    --max_seq_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2
```
The model will be saved in output/lert_qa directory.


## Inference
```bash
bash run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```
