import os
import sys
import json

import glob
import wandb
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Adafactor
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
        '--max_sequence_length',
        type=int,
        default=128,
        help='Max sequence length'
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

    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='Freeze encoder'
    )

    parser.add_argument(
        '--freeze_decoder',
        action='store_true',
        help='Freeze decoder'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )

    parser.add_argument(
        '--accum_iter',
        type=int,
        default=4,
        help='Accumulation iterations'
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

    return t5, tokenizer,


def main(args):

    
    # TODO: Implement multi-gpu training
    # if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    # TODO: Add evaluation metrics (BLEU, BLEU-1, BLEU-2, BLEU-3, BLEU-4, BLEURT, etc.)
    # TODO: Add checkpointing to huggingface

    device = torch.device(args.device)
    model.to(device)


    wandb.init(
        args.wandb_project_name if args.wandb_project_name else None,
    )

    wandb.config.update(args)
    wandb.watch(model)

if __name__ == '__main__':

    args = parse_args()
    main(args)







