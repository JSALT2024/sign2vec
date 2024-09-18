import os
import math
import torch
import evaluate
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5Tokenizer, 
    T5Config,
    Adafactor
)
from sign2vec.model.t5 import T5ModelForSLT
from sign2vec.utils.translation import collate_fn, postprocess_text

from dotenv import load_dotenv
load_dotenv()

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", type=str, default="h2s-test")
    parser.add_argument("--dataset_type", type=str, default="how2sign", choices=["how2sign", "yasl"])
    parser.add_argument("--dataset_dir", type=str, default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe')
    parser.add_argument("--output_dir", default='./results',type=str)
    parser.add_argument("--seed", type=int, default=42)

    # Data processing
    parser.add_argument("--skip_frames", action="store_true")
    parser.add_argument("--max_token_length", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=250)
    parser.add_argument("--transform", type=str, default="yasl", choices=["yasl", "custom"])
    parser.add_argument("--modality", type=str, default="pose", choices=["pose", "sign2vec", "mae"])

    # Training arguments
    parser.add_argument("--embedding_dim", type=int, default=255)
    parser.add_argument("--model_id", type=str, default="t5-small")
    parser.add_argument("--max_training_steps", type=int, default=20_000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=float, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)


    # Running arguments
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--project_name", type=str, default="h2s-t5")
    parser.add_argument("--max_val_samples", type=int, default=None)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Initialize the custom model
    config = T5Config.from_pretrained(args.model_id)
    config.pose_dim = args.embedding_dim  # Dimension of the pose embeddings
    model = T5ModelForSLT(config)

    # Initialize tokenizer and config
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)

    if args.dataset_type == 'how2sign':
        from sign2vec.dataset.how2sign import How2SignForSLT as DatasetForSLT
    elif args.dataset_type == 'yasl':
        from sign2vec.dataset.yasl import YoutubeASLForSLT as DatasetForSLT
    else:
        raise ValueError(f"Dataset type {args.dataset_type} not supported")

    train_dataset = DatasetForSLT(
        h5_fpath=args.dataset_dir,
        mode='train' if not args.dev else 'test',
        transform=args.transform,
        max_token_length=args.max_token_length,
        max_sequence_length=args.max_sequence_length,
        skip_frames=args.skip_frames,
        tokenizer=args.model_id,
        input_type=args.modality,
    )

    if args.dataset_type == 'how2sign':
        val_dataset = DatasetForSLT(
            h5_fpath=args.dataset_dir,
            mode='val' if not args.dev else 'test',
            transform=args.transform,
            max_token_length=args.max_token_length,
            max_sequence_length=args.max_sequence_length,
            skip_frames=args.skip_frames,
            tokenizer=args.model_id,
            max_instances=args.max_val_samples,
            input_type=args.modality,
        )

        test_dataset = DatasetForSLT(
            h5_fpath=args.dataset_dir,
            mode='test',
            transform=args.transform,
            max_token_length=args.max_token_length,
            max_sequence_length=args.max_sequence_length,
            skip_frames=args.skip_frames,
            tokenizer=args.model_id,
            input_type=args.modality,
        )

    elif args.dataset_type == 'yasl':

        val_dataset = DatasetForSLT(
            h5_fpath=args.dataset_dir,
            mode='test',
            transform=args.transform,
            max_token_length=args.max_token_length,
            max_sequence_length=args.max_sequence_length,
            skip_frames=args.skip_frames,
            tokenizer=args.model_id,
            max_instances=args.max_val_samples,
            input_type=args.modality,
        )

        test_dataset = DatasetForSLT(
            h5_fpath=args.dataset_dir,
            mode='test',
            transform=args.transform,
            max_token_length=args.max_token_length,
            max_sequence_length=args.max_sequence_length,
            skip_frames=args.skip_frames,
            tokenizer=args.model_id,
            input_type=args.modality,
        )

    sacrebleu = evaluate.load('sacrebleu')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
            print('Preds:', preds.shape)
            preds = np.argmax(preds, axis=2)
            print('Preds after:', preds.shape)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
        for i in range(50):
            print(f"Prediction: {decoded_preds[i]}")
            print(f"Reference: {decoded_labels[i]}")
            print('*'*50)

        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {
            "bleu": result["score"], 
            'bleu-1': result['precisions'][0],
            'bleu-2': result['precisions'][1],
            'bleu-3': result['precisions'][2],
            'bleu-4': result['precisions'][3],
        }

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    # os.environ["WANDB_DISABLED"] = "true" if not args.dev else "false"

    # Add model and data to device
    model.to("cuda")

    num_train_epochs = args.max_training_steps // (len(train_dataset) // args.per_device_train_batch_size // args.gradient_accumulation_steps)
    num_train_epochs = max(math.ceil(num_train_epochs), 1)

    print(f"""
        Model: {args.model_name}
        Training epochs: {num_train_epochs}
        Number of training steps: {args.max_training_steps}
        Number of training batches: {len(train_dataset) // args.per_device_train_batch_size}
        Number of validation examples: {len(val_dataset)}
    """)

    # Check if total batch size 128
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 128

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=num_train_epochs,
        auto_find_batch_size=True,
        eval_accumulation_steps=1,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        report_to=args.report_to,
        hub_model_id=f"{args.model_name}",
        metric_for_best_model="bleu",
        optim="adafactor",
        hub_token=os.getenv("HUB_TOKEN"),
        
    )

    # Configure the wandb logger
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ["WANDB_NAME"] = args.model_name
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"] = "all"

    import wandb
    wandb.init(
        project=args.project_name, 
        name=args.model_name,
        tags=[args.dataset_type, args.transform, args.modality] + (["dev"] if args.dev else []),
    )

    print('Model device:', training_args.device)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Add safe guard for training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user, saving model...")
        # trainer.save_model()

    print("Running evaluation on test set...")
    # Run inference on the test set
    (logits, _ ), label_ids, eval_results = trainer.predict(test_dataset=test_dataset)
    # Decode the predictions
    predicted_ids = np.argmax(logits, axis=2)
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    # Save the predictions and original sentences to a file
    import json
    with open(os.path.join(args.output_dir, f"{args.model_name}_test_predictions.json"), "w") as f:
        predictions = []
        for idx, pred in enumerate(decoded_preds):
            predictions.append({
                'ground_truth': test_dataset[idx]['sentence'],
                'prediction': pred
            })
        json.dump(predictions, f, indent=4)
        
    # Save evaluation results to json
    with open(os.path.join(args.output_dir, f"{args.model_name}_eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)