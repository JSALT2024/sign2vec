import os
import math
import json
import wandb
import torch
import evaluate
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5Tokenizer, 
    T5Config,
    GenerationConfig,
)
from sign2vec.model.t5 import T5ModelForSLT
from sign2vec.utils.translation import postprocess_text

from dotenv import load_dotenv
load_dotenv()

def init_wandb(args):

    if args.dev:
        os.environ["WANDB_DISABLED"] = "true"
        print("Running in dev mode, disabling wandb")
        return

    wandb.login(
        key=os.getenv("WANDB_API_KEY")
    )
    wandb.init(
        project=args.project_name, 
        name=args.model_name,
        tags=[args.dataset_type, args.transform, args.modality] + (["dev"] if args.dev else []),
    )

    return wandb

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", type=str, default="h2s-test")
    parser.add_argument("--dataset_type", type=str, default="how2sign", choices=["how2sign", "yasl"])
    parser.add_argument("--dataset_dir", type=str, default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe')
    parser.add_argument("--output_dir", default='./results',type=str)
    parser.add_argument("--seed", type=int, default=42)
    
    # New data scheme
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--metadata_file', type=str, required=True)

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
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--pose_dim", type=int, default=208)

    # Evaluation arguments
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)

    # Running arguments
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--project_name", type=str, default="h2s-t5")
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--is_normalized", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    init_wandb(args)

    # Initialize the custom model
    config = T5Config.from_pretrained(args.model_id)
    config.pose_dim = args.pose_dim  # Dimension of the pose embeddings 
    model = T5ModelForSLT(model_name_or_path=args.model_id, config=config)

    # Add collate_fn to DataLoader
    def collate_fn(batch):
        # Add padding to the inputs 
        # "inputs" must be 250 frames long
        # "attention_mask" must be 250 frames long
        # "labels" must be 128 tokens long
        return {
            "sign_inputs": torch.stack([
                torch.cat((sample["sign_inputs"], torch.zeros(args.max_sequence_length - sample["sign_inputs"].shape[0], args.pose_dim)), dim=0)
                for sample in batch
            ]),
            "attention_mask": torch.stack([
                torch.cat((sample["attention_mask"], torch.zeros(args.max_sequence_length - sample["attention_mask"].shape[0])), dim=0)
                if sample["attention_mask"].shape[0] < args.max_sequence_length
                else sample["attention_mask"]
                for sample in batch
            ]),
            "labels": torch.stack([
                torch.cat((sample["labels"].squeeze(0), torch.zeros(args.max_token_length - sample["labels"].shape[0])), dim=0)
                if sample["labels"].shape[0] < args.max_token_length
                else sample["labels"]
                for sample in batch
            ]).squeeze(0).to(torch.long),
        }


    # Initialize tokenizer and config
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)
    model.base_model.generation_config = GenerationConfig(
        max_length=128,
        length_penalty=1.0,
        top_k=10,
        top_p=0.95,
        temperature=1,
        no_repeat_ngram_size=1,
        repetition_penalty=3.1,
        num_beams=3,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        do_sample=True,
    )

    if args.dataset_type == 'how2sign':
        from sign2vec.dataset.how2sign import How2SignForSLT as DatasetForSLT
    elif args.dataset_type == 'yasl':
        from sign2vec.dataset.yasl import YoutubeASLForSLT as DatasetForSLT
    else:
        raise ValueError(f"Dataset type {args.dataset_type} not supported")

    train_dataset = DatasetForSLT(

        h5_fpath=args.dataset_dir,
        mode='train' if not args.dev else 'dev',
        transform=args.transform,
        max_token_length=args.max_token_length,
        max_sequence_length=args.max_sequence_length,
        skip_frames=args.skip_frames,
        tokenizer=args.model_id,
        input_type=args.modality,
        annotation_fpath=args.annotation_file,
        metadata_fpath=args.metadata_file,
    )

    val_dataset = DatasetForSLT(

        h5_fpath=args.dataset_dir,
        mode='dev' if not args.dev else 'dev',
        transform=args.transform,
        max_token_length=args.max_token_length,
        max_sequence_length=args.max_sequence_length,
        skip_frames=args.skip_frames,
        tokenizer=args.model_id,
        max_instances=args.max_val_samples,
        input_type=args.modality,
        annotation_fpath=args.annotation_file,
        metadata_fpath=args.metadata_file,
    )

    # test_dataset = DatasetForSLT(

    #     h5_fpath=args.dataset_dir,
    #     mode='test' if not args.dev else 'dev',
    #     transform=args.transform,
    #     max_token_length=args.max_token_length,
    #     max_sequence_length=args.max_sequence_length,
    #     skip_frames=args.skip_frames,
    #     tokenizer=args.model_id,
    #     input_type=args.modality,
    #     annotation_fpath=args.annotation_file,
    #     metadata_fpath=args.metadata_file,
    # )

    sacrebleu = evaluate.load('sacrebleu')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
            preds = np.argmax(preds, axis=2)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
        for i in range(len(decoded_preds)):
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
    # assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 128

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, args.model_name),
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
        save_strategy="no",
        save_total_limit=3,
        predict_with_generate=True,
        generation_config=GenerationConfig.from_dict({
            "max_length": args.max_sequence_length,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "early_stopping": args.early_stopping,
            "bos_token_id": tokenizer.pad_token_id,
        })
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collate_fn,
    )

    def evaluate(model, dataloader, tokenizer):

        predictions, labels = [], []
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
            outputs = model.generate(
                **batch,
                early_stopping=args.early_stopping,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_length=args.max_sequence_length,
                num_beams=args.num_beams,
                bos_token_id=tokenizer.pad_token_id,
            )

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            predictions.extend(decoded_preds)
            labels.extend([[translation] for translation in decoded_labels])

        return predictions, labels
    
    val_predictions, val_labels = evaluate(model, val_dataloader, tokenizer)

    # Save predictions to file
    with open(os.path.join(args.output_dir, args.model_name, "val_predictions.txt"), "w") as f:
        all_predictions = [
            {
                "prediction": prediction,
                "reference": label[0]
            }
            for prediction, label in zip(val_predictions, val_labels) 
        ]

        json.dump(all_predictions, f)

    val_bleu = sacrebleu.compute(predictions=val_predictions, references=val_labels)

    # Save scores json
    scores = {
        "val": val_bleu,
    }

    with open(os.path.join(args.output_dir, args.model_name, "scores.json"), "w") as f:
        json.dump(scores, f)




