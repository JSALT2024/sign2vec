import os
import sys
import json

import glob
import wandb
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the parent directory to the path so that we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


import torch
import evaluate
from torch import nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer, T5ForConditionalGeneration


from sign2vec.modeling_sign2vec import Sign2VecModel
from finetune.collator import DataCollatorForSign2VecFinetuning

from sign2vec.models.finetune import (
    T5ForSignLanguageTranslation,
    T5BaseForSignLanguageTranslation,
)

from sign2vec.dataset.how2sign_hf5 import How2SignDatasetForFinetuning

def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='Sign2Vec Finetuning')

    parser.add_argument(
        '--experiment_type',
        type=str,
        default='baseline',
        help='Experiment type'
    )

    parser.add_argument(
        '--t5_model_path_or_name', 
        type=str, 
        default='t5-small', 
        help='Model name or path'
    )

    parser.add_argument(
        '--sign2vec_model_path', 
        type=str, 
        default='sign2vec', 
        help='Sign2Vec model path'
    )

    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data', 
        help='Data path'
    )

    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='output', 
        help='Output directory'
    )

    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8, 
        help='Batch size'
    )

    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4, 
        help='Number of workers'
    )

    parser.add_argument(
        '--max_epochs', 
        type=int, 
        default=3, 
        help='Max epochs'
    )

    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=1e-4, 
        help='Learning rate'
    )

    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.01, 
        help='Weight decay'
    )

    parser.add_argument(
        '--warmup_steps', 
        type=int, 
        default=500, 
        help='Warmup steps'
    )

    parser.add_argument(
        '--logging_dir', 
        type=str, 
        default='logs', 
        help='Logging directory'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='dev',
        help='Environment'
    )

    parser.add_argument(
        '--token',
        type=str,
        default='pose',
        help='Token'
    )

    parser.add_argument(
        '--max_frames',
        type=int,
        default=250,
        help='Max frames'
    )

    parser.add_argument(
        '--sign2vec_embed_dim',
        type=int,
        default=1024,
        help='Sign2Vec embed dim'
    )

    parser.add_argument(
        '--t5_embed_dim',
        type=int,
        default=512,
        help='T5 embed dim'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout'
    )

    parser.add_argument(
        '--sampling_rate',
        type=int,
        default=25,
        help='Sampling rate'
    )

    parser.add_argument(
        '--pad_to_multiple_of',
        type=int,
        default=8,
        help='Pad to multiple of'
    )

    parser.add_argument(
        '--mask_time_prob',
        type=float,
        default=0.65,
        help='Mask time probability'
    )

    parser.add_argument(
        '--mask_time_length',
        type=int,
        default=10,
        help='Mask time length'
    )

    parser.add_argument(
        '--wandb_project_name',
        type=str,
        default=None,
        help='Wandb project name'
    )

    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Wandb entity name'
    )

    parser.add_argument(
        '--train_dataset_path',
        type=str,
        default='data/how2sign/dev.csv',
        help='Train dataset path'
    )

    parser.add_argument(
        '--val_dataset_path',
        type=str,
        default='data/how2sign/dev.csv',
        help='Val dataset path'
    )

    parser.add_argument(
        '--test_dataset_path',
        type=str,
        default='data/how2sign/dev.csv',
        help='Test dataset path'
    )


    args = parser.parse_args()

    return args


def get_models(args):
    """Load the models and tokenizers.

    Args:
        args: argparse.Namespace    

    Returns:
        t5: T5ForConditionalGeneration
        tokenizer: AutoTokenizer
        sign2vec: Sign2VecModel
        feature_extractor: Wav2Vec2FeatureExtractor
    """

    tokenizer = AutoTokenizer.from_pretrained(args.t5_model_path_or_name)
    t5 = T5ForConditionalGeneration.from_pretrained(args.t5_model_path_or_name)

    sign2vec = Sign2VecModel.from_pretrained(
        pretrained_model_name_or_path=args.sign2vec_model_path,
        token=args.token,
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=sign2vec.config.input_dim,
        sampling_rate=args.sampling_rate,
        padding_value=0.0,
        do_normalize=True,
    )
    
    return t5, tokenizer, sign2vec, feature_extractor


def main(args):

    t5, tokenizer, sign2vec, feature_extractor = get_models(args)
    
    print('Models loaded!')

    if args.env == 'dev':

        train_dataset = How2SignDatasetForFinetuning(
            dataset=args.train_dataset_path,
            data_dir=args.data_path,
            max_length=args.max_frames,
        )

        val_dataset = How2SignDatasetForFinetuning(
            dataset=args.val_dataset_path,
            data_dir=args.data_path,
            max_length=args.max_frames,
        )

        test_dataset = How2SignDatasetForFinetuning(
            dataset=args.test_dataset_path,
            data_dir=args.data_path,
            max_length=args.max_frames,
        )

        print('Datasets loaded!')

    else:
        raise NotImplementedError('Only dev environment is supported')
    
    
    sample_data = train_dataset[0]
    print('keypoints:', sample_data['input_values'].shape)
    print('sentence:', sample_data['sentence'])
    print('='*50)
    
    
    data_collator = DataCollatorForSign2VecFinetuning(
        model=sign2vec,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=args.pad_to_multiple_of,
        mask_time_prob=args.mask_time_prob,
        mask_time_length=args.mask_time_length,
        tokenizer=tokenizer,
        shift_right=t5._shift_right,
    )

    print('Data collator created!')

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    if args.experiment_type == 'baseline':
        model = T5BaseForSignLanguageTranslation(
            t5_model=t5,
            input_dim=sign2vec.config.input_dim,
            t5_embed_dim=args.t5_embed_dim,
            dropout=args.dropout,
        )

    elif args.experiment_type == 'sign2vec':
        model = T5ForSignLanguageTranslation(
            sign2vec_model=sign2vec,
            t5_model=t5,
            sign2vec_embed_dim=args.sign2vec_embed_dim,
            t5_embed_dim=args.t5_embed_dim,
            dropout=args.dropout,
        )
    else:
        raise NotImplementedError('Only baseline and sign2vec experiments are supported')

    # TODO: Implement multi-gpu training
    # if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    # TODO: Add evaluation metrics (BLEU, BLEU-1, BLEU-2, BLEU-3, BLEU-4, BLEURT, etc.)
    # TODO: Add checkpointing to huggingface

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 5. Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(
        t5.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.warmup_steps,
        gamma=0.1,
    )

    bleu_score = evaluate.load('bleu')

    wandb.init(
        args.wandb_project_name if args.wandb_project_name else None,
    )

    wandb.config.update(args)
    wandb.watch(model)

    # 6. Train the model
    for epoch in range(args.max_epochs):

        total_train_loss = 0
        
        model.train()
        progress_bar = tqdm(total=len(train_loader), desc='Training', leave=False)
        for batch_idx, batch in enumerate(train_loader):

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)

            loss = outputs.loss
            total_train_loss += loss.item()

            wandb.log({
                'train_loss': loss.item(),
            })

            if batch_idx % 200 == 0: 
                
                generated_sentence = []
                for i in range(outputs.shape[0]):
                    generated_sentence.append(
                        tokenizer.decode(outputs[i], skip_special_tokens=True)
                    )

                generated_sentences = tokenizer.batch_decode(
                    
                    skip_special_tokens=False
                )

                decoder_input_ids = batch['decoder_input_ids']
                ground_sentences = tokenizer.batch_decode(
                    decoder_input_ids,
                    skip_special_tokens=True
                )

                for generated_sentence, ground_sentence in zip( generated_sentences, ground_sentences):
                    print('Generated:', generated_sentence)
                    print('References:', ground_sentence)
                    print('='*50)
                
                print(f'epoch: {epoch} | loss: {loss.item()}')

            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.update(1)
    
        total_val_loss = 0

        model.eval()
        instance_count = 0
        bleu_score = evaluate.load('bleu')
        progress_bar = tqdm(total=len(val_loader), desc='Validation', leave=False)
        for batch in val_loader:

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            wandb.log({
                'val_loss': loss.item(),
            })

            generated_sentences = tokenizer.batch_decode(
                model.generate(batch['input_values']),
                skip_special_tokens=False
            )

            decoder_input_ids = batch['decoder_input_ids']
            ground_sentences = tokenizer.batch_decode(
                decoder_input_ids,
                skip_special_tokens=True
            )

            if instance_count < 100:
                for generated_sentence, ground_sentence in zip( generated_sentences, ground_sentences):
                    
                    bleu_score.add_batch(
                        predictions=[generated_sentence], 
                        references=[ground_sentence]
                    )

                    print('Generated:', generated_sentence)
                    print('References:', ground_sentence)
                    print('='*50)

                    instance_count += 1

            total_val_loss += loss.item()
            progress_bar.update(1)
            

        final_score = bleu_score.compute()

        wandb.log({
            'bleu_score': final_score,
        })
        
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {total_train_loss}')

if __name__ == '__main__':

    args = parse_args()
    main(args)







