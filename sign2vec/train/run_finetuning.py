import os
import torch
import evaluate
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5Tokenizer, 
    T5Config
)

from sign2vec.model.t5 import T5ModelForSLT
from sign2vec.dataset.how2sign import How2SignForSLT
from sign2vec.utils.translation import collate_fn, postprocess_text

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", type=str, default="h2s-t5-pose-no-norm")
    parser.add_argument("--h2s_dir", type=str, default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe')
    parser.add_argument("--output_dir", default='./results',type=str)
    parser.add_argument("--seed", type=int, default=42)

    # Data processing
    parser.add_argument("--skip_frames", action="store_true")
    parser.add_argument("--max_token_length", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=250)

    # Training arguments
    parser.add_argument("--model_id", type=str, default="t5-small")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=float, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Initialize the custom model
    config = T5Config.from_pretrained(args.model_id)
    config.pose_dim = 255  # Dimension of the pose embeddings
    model = T5ModelForSLT(config)

    # Initialize tokenizer and config
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)

    # Import How2SignForSLT dataset
    h2s_train = How2SignForSLT(
        h5_fpath=os.path.join(args.h2s_dir, 'H2S_train.h5'),   
        transform=None,
        max_token_length=args.max_token_length,
        max_sequence_length=args.max_sequence_length,
        skip_frames=args.skip_frames,
        tokenizer=args.model_id
    )

    h2s_val = How2SignForSLT(
        h5_fpath=os.path.join(args.h2s_dir, 'H2S_val.h5'),   
        transform=None,
        max_token_length=args.max_token_length,
        max_sequence_length=args.max_sequence_length,
        skip_frames=args.skip_frames,
        tokenizer=args.model_id
    )

    # Import How2SignForSLT dataset
    h2s_test = How2SignForSLT(
        h5_fpath=os.path.join(args.h2s_dir, 'H2S_test.h5'),   
        transform=None,
        max_token_length=args.max_token_length,
        max_sequence_length=args.max_sequence_length,
        skip_frames=args.skip_frames,
        tokenizer=args.model_id
    )

    metric = evaluate.load('sacrebleu')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
            preds = np.argmax(preds, axis=2)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
        for i in range(20):
            print(f"Prediction: {decoded_preds[i]}")
            print(f"Reference: {decoded_labels[i]}")
            print('*'*50)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    # Configure the wandb logger
    os.environ["WANDB_PROJECT"] = "h2s-t5"
    os.environ["WANDB_NAME"] = args.model_name
    os.environ["WANDB_LOG_MODEL"] = "true"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        report_to=args.report_to,
        hub_model_id=f"{args.model_name}",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=h2s_train,
        eval_dataset=h2s_val,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=h2s_test)
    print(eval_results)

    # Save evaluation results to json
    with open(os.path.join(args.output_dir, f"{args.model_name}_eval_results.json"), "w") as f:
        f.write(str(eval_results))