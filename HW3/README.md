Instruction-Tuning Adapter Model : 
CUDA_VISIBLE_DEVICES=0 python model/qlora.py /
    --model_name_or_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo"
    --output_dir /path/to/output 
    --dataset /path/to/training dataset 
    -- dataset_format input-output 

ex CUDA_VISIBLE_DEVICES=1 python model/qlora.py /
    --model_name_or_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo" /--output_dir ./model_result/test /
    --dataset data/train.json /
    --dataset_format input-output /

Checking ppl : 
CUDA_VISIBLE_DEVICES=0 python script/multi_ppl.py /
    --base_model_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo" /
    --checkpoint_dir "model_result/test" /
    --test_data_path "data/public_test.json" /

Inference code: 
CUDA_VISIBLE_DEVICES=0 python script/inference.py /
    --base_model "zake7749/gemma-2-2b-it-chinese-kyara-dpo" /
    -- adapter_model "model_result/test/checkpoint-*** /
    --input_file "data/public_test.json" /
    --output_file "prediction.json"