import argparse 
import csv 
import datasets
import json 
import os 
import torch 
import numpy as np
import logging

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from itertools import chain
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    default_data_collator,
)
from utils_qa import postprocess_qa_predictions

logger = get_logger(__name__)

def load_model_and_tokenizer(model_path, device):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(device)
    return model, tokenizer

def load_contexts(context_file):
    with open(context_file, "r", encoding="utf-8") as f:
        return json.load(f)

def load_questions(question_file):
    with open(question_file, "r", encoding="utf-8") as f:
        return json.load(f)
    
def answer_question(model , tokenizer , context ,question , device):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True
    )
    pad_on_right = tokenizer.padding_side == "right"

    inputs = {k:v.to(device) for k,v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]
    max_length = 512 if input_length > 512 else 512

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        trucation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    inputs = {k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    print("start:" + str(answer_start))
    print("end:" + str(answer_end))
    if answer_start == 0 and answer_end == 1:
        return None
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer 

def save_results(results, output_file):
    with open(output_file, "w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"結果已保存到 {output_file}")

def main(args):
    accelerator_log_kwargs = {}
    accelerator = Accelerator(**accelerator_log_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    model,tokenizer = load_model_and_tokenizer(args.model_path, device)

    contexts = load_contexts(args.context_file)
    questions = load_questions(args.questions_file)

    pad_on_right = tokenizer.padding_side == "right"
    raw_datasets = datasets.Dataset.from_list(questions)

    max_seq_length = min(512,tokenizer.model_max_length)

    def process_dataset(example):
        example["context"] = contexts[example["predicted_relevant"]]
        return example
    
    raw_datasets = raw_datasets.map(
        process_dataset,
        batched=False,
        remove_columns=["predicted_relevant"]
    )
    
    def prepare_test_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]
        
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    with accelerator.main_process_first():
        tokenized_test_datasets = raw_datasets.map(
            prepare_test_features,
            batched=True,
            remove_columns=raw_datasets.column_names,
        )
    
    test_dataset_for_model = tokenized_test_datasets.remove_columns(["example_id", "offset_mapping"])

    test_dataloader = DataLoader(
        test_dataset_for_model,
        collate_fn=default_data_collator,
        batch_size=8,
        shuffle=False,
    )

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir=None,
            prefix=stage,
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        return formatted_predictions

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat


    model.to(device)
    # Prepare everything with our `accelerator`.
    (
        model,
        test_dataloader,
    ) = accelerator.prepare(model, test_dataloader)

    # Start testing
    logger.info("***** Running span selection *****")
    logger.info(f"  Num examples = {len(raw_datasets)}")
    progress_bar = tqdm(range(len(test_dataloader)))
    progress_bar.update(0)

    # Start prediction
    all_start_logits = []
    all_end_logits = []

    model.eval()
    for idx, input in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(**input)
        start_logits = output.start_logits
        end_logits = output.end_logits

        all_start_logits.append(start_logits.cpu().numpy())
        all_end_logits.append(end_logits.cpu().numpy())
        progress_bar.update(1)

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, tokenized_test_datasets, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, tokenized_test_datasets, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    all_predictions = post_processing_function(raw_datasets, tokenized_test_datasets, outputs_numpy)
    with open(args.output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "answer"])
        for prediction in all_predictions:
            writer.writerow([prediction["id"], prediction["prediction_text"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="問答模型")
    parser.add_argument("--model_path", type=str, default="output_qa", help="模型的路徑")
    parser.add_argument("--context_file", type=str, default="context.json", help="上下文檔案的路徑")
    parser.add_argument("--questions_file", type=str, default="test_results.json", help="問題檔案的路徑")
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument("--output_file", type=str, default="qa_result_new.csv", help="結果輸出的檔案路徑")

    args = parser.parse_args()
    main(args)