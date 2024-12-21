from datasets import load_dataset 
from transformers import AutoTokenizer 
from transformers import DataCollatorForSeq2Seq
from transformers import AdamW
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import EarlyStoppingCallback
import evaluate 
import numpy as np 
import torch 
import os 
from transformers import AutoModelForSeq2SeqLM , Seq2SeqTrainingArguments, Seq2SeqTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using GPU/CPU device :{device}")

print(f"Current CUDA device:{torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# 獲取腳本所在目錄的絕對路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir,'..','data','train.jsonl')

def load_and_process_data(path):
    dataset = load_dataset('json',data_files = path)
    # 移除多餘列，留下maintext、title 
    dataset = dataset.remove_columns(['date_publish','source_domain','split','id'])
    # 分割數據集 
    train_valid_split = dataset['train'].train_test_split(test_size=0.2)

    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']
    
    return train_dataset , valid_dataset



def main():

    train_dataset, valid_dataset = load_and_process_data(data_path)

    checkpoint = "google/mt5-small"
    tokenizer = MT5Tokenizer.from_pretrained(checkpoint)

    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["maintext"]]
        model_inputs = tokenizer(inputs, max_length = 1024 , truncation = True,padding='max_length',return_tensors='pt')

        labels = tokenizer(text_target = examples["title"], max_length = 128,truncation = True,padding='max_length',return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        return model_inputs

    train_object = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    valid_object = valid_dataset.map(preprocess_function, batched=True, remove_columns=valid_dataset.column_names)
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = checkpoint)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions , labels = eval_pred
        predictions = np.where(predictions != -100,predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens = True) 
        labels = np.where(labels != -100 , labels , tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels,skip_special_tokens=True) 
        
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge.compute(predictions = decoded_preds, references = decoded_labels,rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer = True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    model = MT5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    torch.save(model.state_dict(), 'model_weights.pth')

    training_args = Seq2SeqTrainingArguments(
        output_dir="./model_result/train",
        evaluation_strategy="epoch",
        learning_rate = 3e-04,
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay = 0.01,
        save_total_limit = 3,
        num_train_epochs = 20,
        predict_with_generate=True,
        gradient_accumulation_steps=3,
        fp16 = False, # change to bf16=True forXPU 
        max_grad_norm=1.0,
        push_to_hub=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,  # 添加這行
        metric_for_best_model="eval_loss",  # 添加這行
        greater_is_better=False,  # 添加這行
    )

    optimizer = AdamW(model.parameters(), lr=5e-05)
    # 創建 EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # 如果連續3個 epoch 沒有改善，就停止訓練
        early_stopping_threshold=0.01  # 改善需要超過這個閾值才算
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args = training_args,
        train_dataset=train_object,
        eval_dataset=valid_object,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[early_stopping_callback]  # 添加早停回調
    )
     

    """ def predict_with_beams(trainer, num_beams=8):
        predictions = trainer.predict(valid_object, num_beams=num_beams)
        return predictions """

    print(f"Model is on device: {model.device}")
    torch.cuda.empty_cache()

    # 调用 predict_with_beams 使用 num_beams
    """ predict_with_beams(trainer, num_beams=4) """
    trainer.train()

if __name__ == '__main__':
    main()

