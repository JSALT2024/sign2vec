import os
import json
import torch
import argparse
import numpy as np
from dotenv import load_dotenv
from transformers import T5Tokenizer
from sign2vec.model.configuration_t5 import SignT5Config
from sign2vec.model.modeling_t5 import T5ModelForSLT
from sign2vec.utils.translation import postprocess_text

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned T5 model for SLT")

    # Model and data paths
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing the fine-tuned model and config.")
    parser.add_argument("--model_name", type=str, default="h2s-test", help="Model name or folder inside model_dir.")
    parser.add_argument("--dataset_type", type=str, default="how2sign", choices=["how2sign", "yasl"], help="Type of the dataset.")
    parser.add_argument("--dataset_dir", type=str, default="/path/to/data", help="Path to the dataset directory.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotations file.")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to the metadata file.")

    # Generation parameters
    parser.add_argument("--max_sequence_length", type=int, default=250, help="Max number of frames for sign inputs.")
    parser.add_argument("--max_token_length", type=int, default=128, help="Max token length for labels.")
    parser.add_argument("--pose_dim", type=int, default=208, help="Dimension of the pose embeddings.")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search.")
    parser.add_argument("--length_penalty", type=float, default=0.6, help="Length penalty for generation.")
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping in generation.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No repeat ngram size.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--input_type", type=str, default="pose", choices=["pose", "sign2vec", "mae"], help="Input modality.")

    # Custom arguments
    parser.add_argument("--is_normalized", action="store_true", help="If the data is normalized.")
    parser.add_argument("--transform", type=str, default="yasl", choices=["yasl", "custom"], help="Data transform type.")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="File to store predictions.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    parser.add_argument("--dev", action="store_true", help="Use dev mode.")
    
    return parser.parse_args()

def collate_fn(batch, max_sequence_length, max_token_length, pose_dim):
    return {
        "sign_inputs": torch.stack([
            torch.cat((sample["sign_inputs"], torch.zeros(max_sequence_length - sample["sign_inputs"].shape[0], pose_dim)), dim=0)
            for sample in batch
        ]),
        "attention_mask": torch.stack([
            torch.cat((sample["attention_mask"], torch.zeros(max_sequence_length - sample["attention_mask"].shape[0])), dim=0)
            if sample["attention_mask"].shape[0] < max_sequence_length
            else sample["attention_mask"]
            for sample in batch
        ]),
        "labels": torch.stack([
            torch.cat((sample["labels"].squeeze(0), torch.zeros(max_token_length - sample["labels"].shape[0])), dim=0)
            if sample["labels"].shape[0] < max_token_length
            else sample["labels"]
            for sample in batch
        ]).squeeze(0).to(torch.long),
    }

def load_dataset(args):
    if args.dataset_type == 'how2sign':
        from sign2vec.dataset.how2sign import How2SignForSLT as DatasetForSLT
        mode = 'test' if not args.dev else 'dev'
    elif args.dataset_type == 'yasl':
        from sign2vec.dataset.yasl import YoutubeASLForSLT as DatasetForSLT
        mode = 'dev' if args.dev else 'test'
    else:
        raise ValueError(f"Dataset type {args.dataset_type} not supported")

    dataset = DatasetForSLT(
        h5_fpath=args.dataset_dir,
        mode=mode,
        transform=args.transform,
        max_token_length=args.max_token_length,
        max_sequence_length=args.max_sequence_length,
        skip_frames=False,
        tokenizer=args.model_dir,
        input_type=args.input_type,
        annotation_fpath=args.annotation_file,
        metadata_fpath=args.metadata_file,
        is_normalized=args.is_normalized,
        verbose=args.verbose,
    )
    return dataset

def evaluate(model, dataloader, tokenizer, args):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
            outputs = model.generate(
                **batch,
                early_stopping=args.early_stopping,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_length=args.max_sequence_length,
                num_beams=args.num_beams,
                bos_token_id=tokenizer.pad_token_id,
                length_penalty=args.length_penalty,
            )
            # Replace invalid tokens with <unk>
            if len(np.where(outputs.cpu().numpy() > len(tokenizer) - 1)[1]) > 0 and args.verbose:
                print(f'Replacing <unk> for illegal tokens found on indexes {np.where(outputs > len(tokenizer) - 1)[1]}')
            outputs[outputs > len(tokenizer) - 1] = tokenizer.unk_token_id

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            predictions.extend(decoded_preds)
            labels.extend([[translation] for translation in decoded_labels])
    return predictions, labels

def main():
    args = parse_args()

    # Load model and tokenizer
    config = SignT5Config(
        base_model_name=os.path.join(args.model_dir, args.model_name),
        sign_input_dim=args.pose_dim,
    )
    model = T5ModelForSLT.from_pretrained(os.path.join(args.model_dir, args.model_name), config=config)
    tokenizer = T5Tokenizer.from_pretrained(os.path.join(args.model_dir, args.model_name))

    # Prepare dataset
    dataset = load_dataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(
            batch,
            max_sequence_length=args.max_sequence_length,
            max_token_length=args.max_token_length,
            pose_dim=args.pose_dim
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions, labels = evaluate(model, dataloader, tokenizer, args)

    # Postprocess predictions and references
    decoded_preds, decoded_labels = postprocess_text(predictions, [ref[0] for ref in labels])

    if args.verbose:
        for i in range(min(5, len(decoded_preds))):
            print("Prediction:", decoded_preds[i])
            print("Reference:", decoded_labels[i])
            print("-" * 50)

    # Save predictions
    all_predictions = [
        {
            "prediction": pred,
            "reference": ref
        }
        for pred, ref in zip(decoded_preds, decoded_labels)
    ]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()