import os
import sys
import json
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the parent directory to the path so that we can import the modules
sys.path.append(os.path.dirname('../'))

import torch
import evaluate
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer, T5ForConditionalGeneration


from sign2vec.modeling_sign2vec import Sign2VecModel
from pretraining.utils.config import Sign2VecConfig
from pretraining.utils.collator import DataCollatorForSign2VecPretraining

from sign2vec.models.finetune import T5ForSignLanguageTranslation
from sign2vec.dataset.how2sign import How2SignDatasetForPretraining, get_how2sign_dataset

def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='Sign2Vec Finetuning')

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

        train_df, val_df, test_df = get_how2sign_dataset(
            DATASET_PATH=args.data_path,
            verbose=True
        )

        train_dataset = How2SignDatasetForPretraining(
            dataframe=train_df,
            keypoint_path=args.data_path,
            tokenizer=tokenizer,
            max_frames=args.max_frames
        )

        val_dataset = How2SignDatasetForPretraining(
            dataframe=val_df,
            keypoint_path=args.data_path,
            tokenizer=tokenizer,
            max_frames=args.max_frames
        )

        test_dataset = How2SignDatasetForPretraining(
            dataframe=test_df,
            keypoint_path=args.data_path,
            tokenizer=tokenizer,
            max_frames=args.max_frames
        )

        print('Datasets loaded!')

    else:
        raise NotImplementedError('Only dev environment is supported')
    
    
    data_collator = DataCollatorForSign2VecPretraining(
        model=sign2vec,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=args.pad_to_multiple_of,
        mask_time_prob=args.mask_time_prob,
        mask_time_length=args.mask_time_length,
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

    print('='*50)
    sample_data = next(iter(val_loader))
    print('keypoints:', sample_data['input_values'].shape)
    print('mask_time_indices:', sample_data['mask_time_indices'].shape)
    print('sampled_negative_indices:', sample_data['sampled_negative_indices'].shape)
    print('='*50)

    print('DataLoaders created!')

    model = T5ForSignLanguageTranslation(
        sign2vec_model=sign2vec,
        t5_model=t5,
        sign2vec_embed_dim=args.sign2vec_embed_dim,
        t5_embed_dim=args.t5_embed_dim,
        dropout=args.dropout,
    )

    # TODO: Implement multi-gpu training
    # if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    # TODO: Add evaluation metrics (BLEU, BLEU-1, BLEU-2, BLEU-3, BLEU-4, BLEURT, etc.)
    # TODO: Add checkpointing to huggingface
    # TODO: Add wandb logging


    # model.to('cuda')
    
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

    # import wandb
    # wandb.init(
    #     project=args.wandb_project_name if args.wandb_project_name else None,
    #     entity=args.wandb_run_name if args.wandb_run_name else None,
    # )

    # 6. Train the model
    for epoch in range(args.max_epochs):

        total_train_loss = 0
        total_train_score = 0
        
        model.train()
        for batch in tqdm(test_loader, desc='Training', leave=False):
            optimizer.zero_grad()
            outputs = model(**batch)

            loss = outputs.loss
            total_train_loss += loss.item()
            total_train_score += outputs.score

            # wandb.log({
            #     'train_loss': loss.item(),
            #     # 'train_score': outputs.score,
            # })

            loss.backward()
            optimizer.step()
            scheduler.step()
    
        total_val_loss = 0
        total_val_score = 0

        model.eval()
        for batch in val_loader:
            outputs = model(**batch)
            loss = outputs.loss

            # wandb.log({
            #     'val_loss': loss.item(),
            #     # 'val_score': outputs.score,
            # })



            total_val_loss += loss.item()
            total_val_score += outputs.score

        
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {total_train_loss}')

if __name__ == '__main__':

    args = parse_args()
    main(args)







