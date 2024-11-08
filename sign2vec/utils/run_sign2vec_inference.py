import os
import h5py
import torch
from torch.utils.data import Dataset
from sign2vec.utils.config import Sign2VecConfig
from sign2vec.model.modeling_sign2vec import Sign2VecModel

from dataclasses import dataclass
from typing import List, Dict, Union, Optional
from sign2vec.model.modeling_sign2vec import (
    Sign2VecForPreTraining, 
    Sign2VecFeatureExtractor,
    _compute_mask_indices,
    _sample_negative_indices
)
from torch.nn.utils.rnn import pad_sequence

@dataclass
class DataCollatorForSign2VecPretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Sign2VecForPreTraining`):
            The Sign2Vec model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Sign2VecFeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: Sign2VecForPreTraining
    feature_extractor: Sign2VecFeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = {}
        # Features are [(clip_id, pose, sentence), ...] where pose is a list of features
        batch.update({
            'sentences': [item[2] for item in features],
            'clip_ids': [item[0] for item in features],
        })

        batch['input_values'] = pad_sequence([torch.tensor(item[1]) for item in features], batch_first=True)

        # NOTE: transpose input_values to have <POSE> dimension first
        batch["input_values"] = batch["input_values"].transpose(1, 2)

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length) if self.model.config.encoder_type == 'conv_layer' else batch["input_values"].shape[-1]
        
        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        return batch


def parse_args():

    import argparse
    parser = argparse.ArgumentParser('Run Sign2Vec Inference on a given dataset')

    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="id of dataset"
    )    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="Type of the dataset",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the config file",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="pose",
        help="Type of the input modality",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="yasl",
        help="Type of the input transformation",
    )
    parser.add_argument(
        "--zero_mean",
        action="store_true",
        help="Zero mean the input features",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run on the dev dataset"
    )
    parser.add_argument(
        "--min_sequence_length",
        type=int,
        default=100,
        help="Minimum sequence length for the input features",
    )
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Initialize the custom model
    if args.model_id is not None:
        # model = Sign2VecModel.from_pretrained(args.model_id)
        config = Sign2VecConfig().load_from_yaml(yaml_file=args.config_file)
        model = Sign2VecModel(config)


    print(f"Loading model from {args.model_id}")
    print(model)

    # Initialize dataset
    if args.dataset_type == 'how2sign':
        from sign2vec.dataset.how2sign import How2SignForPose as DatasetForPose
        train_dataset = DatasetForPose(
            h5_fpath=args.dataset_dir + ("/H2S_train.h5" if not args.dev else "/H2S_test.h5"),
            transform=args.transform,
            zero_mean=args.zero_mean,
        )
        val_dataset = DatasetForPose(
            h5_fpath=args.dataset_dir + ("/H2S_val.h5" if not args.dev else "/H2S_test.h5"),
            transform=args.transform,
            zero_mean=args.zero_mean,
        )
        test_dataset = DatasetForPose(
            h5_fpath=args.dataset_dir + "/H2S_test.h5",
            transform=args.transform,
            zero_mean=args.zero_mean,
        )
    elif args.dataset_type == 'yasl':
        from sign2vec.dataset.yasl import YoutubeASLForPose as DatasetForPose
    else:
        raise ValueError(f"Dataset type {args.dataset_type} not supported")
    
    feature_extractor = Sign2VecFeatureExtractor(config=model.config)
    collator = DataCollatorForSign2VecPretraining(
        model=model,
        feature_extractor=feature_extractor,
    )
    # Run inference on the datasets 
    model.eval()
    for mode, dataset in [
            ('train', train_dataset), 
            ('val', val_dataset), 
            ('test', test_dataset)
        ]:
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collator
        )
        
        fname = '.'.join([
            ('H2S' if args.dataset_id == 'h2s' else 'YouTubeASL'),
            'sign2vec',
            mode,
            ('0' if args.dataset_id == 'yasl' else ''),
            'h5'
        ])

        with h5py.File(args.save_dir + fname, "a") as h5file:

            for idx in range(len(dataloader)):
                
                batch = next(iter(dataloader))

                sentences = batch.get('sentences')
                clip_ids = batch.get('clip_ids')

                batch.pop('sentences')
                batch.pop('clip_ids')
                
                with torch.no_grad(): features = model.forward(**batch, return_dict=True)
                
                print('Model out shape:', features.last_hidden_state.shape)
                for idx, (sentence, clip_id) in enumerate(zip(
                    sentences, clip_ids
                )):
                    
                    print(
                        'Clip_ids:',
                        clip_id,
                        'Input_shape:',
                        batch['input_values'][idx].shape,
                        '**Output Features:',
                        features.last_hidden_state[idx].shape,
                        'Has nans',
                        features.last_hidden_state[idx].nansum(),
                        '**Extract Features:',
                        features.last_hidden_state[idx],
                        '*************************',
                        sep='\n'
                    )

                    # print(f"Saving {clip_id} ...")
                    # h5file.create_group(clip_id)
                    # h5file[clip_id].create_dataset("features", data=features.last_hidden_state)
                    # h5file[clip_id].create_dataset("sentence", data=sentence)
