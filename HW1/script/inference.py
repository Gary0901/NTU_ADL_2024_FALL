import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoConfig, AutoModelForQuestionAnswering, default_data_collator
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from utils_qa import postprocess_qa_predictions

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple choice and question answering models")
    parser.add_argument("--mq_model_dir", type=str, required=True, help="Directory containing multiple choice model files")
    parser.add_argument("--qa_model_dir", type=str, required=True, help="Directory containing question answering model files")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test.json file")
    parser.add_argument("--context_file", type=str, required=True, help="Path to the context.json file")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--pretrained_model", type=str, default="hfl/chinese-lert-base", help="Name or path of the pretrained model")
    parser.add_argument("--output_file", type=str, default="final_output.csv", help="Path for the final output CSV file")
    parser.add_argument("--version_2_with_negative", action="store_true", help="If true, some of the examples do not have an answer")
    parser.add_argument("--doc_stride", type=int, default=128, help="Stride for document processing in QA model")
    return parser.parse_args()

def preprocess_function_mq(examples, tokenizer, max_length, context_dict):
    questions = [example["question"] for example in examples]
    paragraphs = [example["paragraphs"] for example in examples]
    
    first_sentences = [[question] * 4 for question in questions]
    second_sentences = [[context_dict[p] for p in para] for para in paragraphs]
    
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

def run_multiple_choice_model(args, test_data, context_dict):
    accelerator = Accelerator()
    device = accelerator.device

    config = AutoConfig.from_pretrained(args.mq_model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(args.mq_model_dir, config=config)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
        logger.info(f"Loaded tokenizer from config.name_or_path: {config.name_or_path}")
    except Exception as e:
        logger.info(f"Failed to load tokenizer from config.name_or_path: {e}")
        logger.info(f"Attempting to load tokenizer from specified pretrained model: {args.pretrained_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    processed_test_data = preprocess_function_mq(test_data, tokenizer, args.max_seq_length, context_dict)

    test_dataset = [{"input_ids": torch.tensor(input_ids), 
                     "attention_mask": torch.tensor(attention_mask)} 
                    for input_ids, attention_mask in zip(processed_test_data["input_ids"], processed_test_data["attention_mask"])]
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Running Multiple Choice Model"):
            outputs = model(**batch)
            predictions.extend(accelerator.gather(outputs.logits).argmax(dim=-1).cpu().tolist())

    for i, pred in enumerate(predictions):
        test_data[i]["predicted_relevant"] = test_data[i]["paragraphs"][pred]

    return test_data

def prepare_qa_features(examples, tokenizer, max_seq_length, doc_stride):
    pad_on_right = tokenizer.padding_side == "right"
    examples["question"] = [q.lstrip() for q in examples["question"]]
    
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def run_question_answering_model(args, mq_output, context_dict):
    accelerator = Accelerator()
    device = accelerator.device

    model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.qa_model_dir)

    for item in mq_output:
        item["context"] = context_dict[item["predicted_relevant"]]

    raw_datasets = datasets.Dataset.from_list(mq_output)
    
    tokenized_datasets = raw_datasets.map(
        lambda examples: prepare_qa_features(examples, tokenizer, args.max_seq_length, args.doc_stride),
        batched=True,
        remove_columns=raw_datasets.column_names,
    )

    data_collator = default_data_collator
    eval_dataset = tokenized_datasets.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()

    all_start_logits = []
    all_end_logits = []

    for batch in tqdm(eval_dataloader, desc="Question Answering"):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = accelerator.gather(outputs.start_logits).cpu().numpy()
            end_logits = accelerator.gather(outputs.end_logits).cpu().numpy()
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = np.concatenate(all_start_logits, axis=0)
    end_logits_concat = np.concatenate(all_end_logits, axis=0)
    
    outputs = (start_logits_concat, end_logits_concat)
    
    theoretical_answers = postprocess_qa_predictions(
        raw_datasets,
        tokenized_datasets,
        outputs,
        args.version_2_with_negative,
        20,
        30,
        0.0,
    )
    
    return theoretical_answers

def main():
    args = parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device
    
    logger.info(f"Using device: {device}")

    with open(args.context_file, "r", encoding="utf-8") as f:
        context_dict = json.load(f)

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info("Running Multiple Choice Model...")
    mq_output = run_multiple_choice_model(args, test_data, context_dict)

    logger.info("Running Question Answering Model...")
    qa_output = run_question_answering_model(args, mq_output, context_dict)

    logger.info(f"Saving final output to {args.output_file}")
    with open(args.output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "answer"])
        for id, answer in qa_output.items():
            writer.writerow([id, answer])

if __name__ == "__main__":
    main()