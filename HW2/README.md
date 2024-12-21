train the model 
CUDA_VISIBLE_DEVICES=0 python train.py 
the model will be saved in './model_result/train' 
but make sure the training dataset path is correct

inferece the model 
CUDA_VISIBLE_DEVICES=0 python inference.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--model_path MODEL_PATH] [--strategy STRATEGY]
like python inference.py --input data/public.jsonl --output output/output.jsonl --model_path model_result/train --strategy beam_search