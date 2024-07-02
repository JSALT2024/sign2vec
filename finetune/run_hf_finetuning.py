import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from t5 import CustomT5Model
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)

# Add the parent directory to the path so that we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sign2vec.dataset.how2sign_hf5 import How2SignDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune T5 model for Sign Language Translation")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the model")
    parser.add_argument("--t5_model_path_or_name", type=str, required=True, help="Path or name of the T5 model")


    parser.add_argument("--debug", action="store_true", help="Run the script in debug mode")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the input sequence")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")

    # training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Total number of checkpoints to save")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")

    return parser.parse_args()


args = parse_args()

# 1. Setup Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.t5_model_path_or_name)
model = CustomT5Model.from_pretrained(args.t5_model_path_or_name).to(device)
loader = How2SignDataset

train_df = pd.read_csv(args.train_data_path)
val_df = pd.read_csv(args.val_data_path)
test_df = pd.read_csv(args.test_data_path)


if args.debug:
    train_df = train_df.sample(100)
    val_df = val_df.sample(100)
    test_df = test_df.sample(100)

train_dataset = Dataset.from_dict(train_df.to_dict(orient="list"))
val_dataset = Dataset.from_dict(val_df.to_dict(orient="list"))
# test_dataset = Dataset.from_dict(test_df.to_dict(orient="list"))


def preprocess_pose(h5_file_path, sentence_idx):
    h5_path = os.path.join( args.data_dir , h5_file_path )
    dataset = loader(h5_path)

    data, sentence = dataset.load_data(idx=sentence_idx)
    
    pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data

    face_landmarks = face_landmarks.reshape(-1, 32*2)
    pose_landmarks = pose_landmarks.reshape(-1, 7*2)
    right_hand_landmarks = right_hand_landmarks.reshape(-1, 21*2)
    left_hand_landmarks = left_hand_landmarks.reshape(-1, 21*2)
    
    data = np.concatenate([pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks], axis=1)

    data = torch.tensor(data).reshape(data.shape[0], -1)
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data


def preprocess_function(examples):    
    targets = examples['sentence']
    labels = tokenizer(targets, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt")
    file_paths = examples['h5_file_path']
    continuous_input = preprocess_pose(file_paths, examples['sentence_idx'])
    return ({ 
        "labels": labels["input_ids"],
        "continuous_input": continuous_input,
    })


# 4. Data Collator
class CustomDataCollator:
    def __call__(self, batch):
        labels = [torch.tensor(example['labels'], dtype=int) for example in batch]
        labels = torch.stack(labels)

        continuous_input = pad_sequence([
            torch.tensor(example['continuous_input']) for example in batch
        ], batch_first=True, padding_value=0.0)

        # crop the continuous input to the maximum length
        continuous_input = continuous_input[:, :args.max_length, :]


        attention_mask = torch.ones(
            continuous_input.shape[0],
            continuous_input.shape[1]
        )  # create an attention mask for the continuous input

        return {
            'labels': labels, 
            'continuous_input': continuous_input, 
            'attention_mask': attention_mask,
            'decoder_input_ids': labels
        }
    
os.environ["WANDB_DISABLED"] = "true"

print(" *** Using device:", device)

# 5. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    weight_decay=args.weight_decay,
    save_total_limit=args.save_total_limit,
    num_train_epochs=args.num_train_epochs,
    logging_dir=args.logging_dir,
    push_to_hub=False,
    report_to=None
)

# 3. Prepare Dataset
print("Preparing datasets...")
train_dataset = train_dataset.map(preprocess_function)
val_dataset = val_dataset.map(preprocess_function)
# test_dataset = test_dataset.map(preprocess_function)
data_collator = CustomDataCollator()

metric = load_metric("sacrebleu")

import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# 6. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()


# Save the model
model.save_pretrained("./custom_t5_translation_model")
tokenizer.save_pretrained("./custom_t5_translation_model")

