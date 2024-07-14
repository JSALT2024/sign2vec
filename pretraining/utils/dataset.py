import datasets
from utils.bobsl import BOBSLDataset
from transformers import Wav2Vec2FeatureExtractor
from utils.collator import DataCollatorForWav2Vec2Pretraining
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from sign2vec.dataset.how2sign_hf5 import How2SignDatasetForPretraining, YoutubeASLForPretraining
from sign2vec.modeling_sign2vec import Sign2VecFeatureEncoder

channel_size = {
    'face_keypoints_2d': 70,
    'hand_left_keypoints_2d': 21,
    'hand_right_keypoints_2d': 21,
    'pose_keypoints_2d': 25,
}

def prepare_dataloader(args, config, model, accelerator):
    
    # load the dataset

    sampling_rate = 25

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=config.input_dim,
        sampling_rate=sampling_rate,
        do_normalize=False,
    )

    # set max & min audio length in number of samples
    max_length = int(args.max_duration_in_seconds * sampling_rate)
    min_length = int(args.min_duration_in_seconds * sampling_rate)

    # load via mapped files via path
    cache_file_names = None
    if args.train_cache_file_name is not None: 
        cache_file_names = {
            "train": args.train_cache_file_name, 
            "validation": args.validation_cache_file_name
        }

    # load audio files into numpy arrays
    with accelerator.main_process_first():
        if args.dataset_name == 'how2sign':
            vectorized_datasets = {
                "train": How2SignDatasetForPretraining(
                    dataset=args.train_data_path,
                    data_dir=args.data_dir,
                    max_length=max_length,
                ),
                "validation": How2SignDatasetForPretraining(
                    dataset=args.validation_data_path,
                    data_dir=args.data_dir,
                    max_length=max_length,
                )
            }
        elif args.dataset_name == 'yasl':
            vectorized_datasets = {
                "train": YoutubeASLForPretraining(
                    dataset=args.train_data_path,
                    data_dir=args.data_dir,
                    max_length=max_length,
                    kp_norm=args.kp_norm,
                    zero_mean_unit_var_norm=args.zero_mean_unit_var_norm,
                    add_noise=args.add_noise,
                ),
                "validation": YoutubeASLForPretraining(
                    dataset=args.validation_data_path,
                    data_dir=args.data_dir,
                    max_length=max_length,
                    kp_norm=args.kp_norm,
                    zero_mean_unit_var_norm=args.zero_mean_unit_var_norm,
                    add_noise=args.add_noise,
                )
            }

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    mask_time_prob = config.mask_time_prob if args.mask_time_prob is None else args.mask_time_prob
    mask_time_length = config.mask_time_length if args.mask_time_length is None else args.mask_time_length

    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
    )

    train_dataloader = DataLoader(
        vectorized_datasets["train"],
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
    )

    print('DataLoader created successfully!')

    eval_dataloader = DataLoader(
        vectorized_datasets["validation"], 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
        shuffle=False
    )

    print('DataLoader created successfully!')

    print('Total Number of Training Instances', len(vectorized_datasets["train"]))

    test = next(iter(train_dataloader))

    print(test['input_values'][0])
    print(test['input_values'].shape)
    print(test['mask_time_indices'].shape)

    batch_size, channels, raw_sequence_length = test['input_values'].shape
    sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()

    print(f'Number of discrete tokens per {args.max_duration_in_seconds} seconds video:', raw_sequence_length)
    print('====================')

    return train_dataloader, eval_dataloader