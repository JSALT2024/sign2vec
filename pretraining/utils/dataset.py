import datasets
from utils.bobsl import BOBSLDataset
from transformers import Wav2Vec2FeatureExtractor
from utils.collator import DataCollatorForWav2Vec2Pretraining
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data.dataloader import DataLoader

from ..sign2vec.modeling_sign2vec import Sign2VecFeatureEncoder

channel_size = {
    'face_keypoints_2d': 70,
    'hand_left_keypoints_2d': 21,
    'hand_right_keypoints_2d': 21,
    'pose_keypoints_2d': 25,
}

def prepare_dataloader(args, config, model, accelerator):
    
    # 1. Set the correct target sampling rate
    sampling_rate = int(
        channel_size['pose_keypoints_2d'] * 2 +
        channel_size['face_keypoints_2d'] * 2 + 
        channel_size['hand_left_keypoints_2d'] * 2  +
        channel_size['hand_right_keypoints_2d'] * 2 
    ) * args.fps

    print('Sampling rate:', sampling_rate)

    # 2. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
    feature_extractor = Wav2Vec2FeatureExtractor(config)

    # only normalized-inputs-training is supported
    # if not feature_extractor.do_normalize:
    #     raise ValueError(
    #         "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
    #     )

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
        vectorized_datasets = {
            'train': BOBSLDataset(
                data_path=args.train_data_path,
                info_path=args.train_info_path,
                use_face=args.use_face,
                use_hands=args.use_hands,
                use_pose=args.use_pose,
                stride=int(args.stride),
                max_length=max_length,
                sampling_rate=sampling_rate,
                feature_extractor=feature_extractor,
            ),
            'validation': BOBSLDataset(
                data_path=args.validation_data_path,
                info_path=args.validation_info_path,
                use_face=args.use_face,
                use_hands=args.use_hands,
                use_pose=args.use_pose,
                stride=int(args.stride),
                max_length=max_length,
                sampling_rate=sampling_rate,
                feature_extractor=feature_extractor,
            ),
        }

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if args.preprocessing_only:
        return


    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. Define data collator, optimizer and scheduler

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
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        vectorized_datasets["validation"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    print('DataLoader created successfully!')

    test = next(iter(train_dataloader))

    print(test['input_values'].shape)
    print(test['mask_time_indices'].shape)

    batch_size, raw_sequence_length = test['input_values'].shape
    sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()

    print(f'Number of discrete tokens per {args.max_duration_in_seconds} seconds video:', sequence_length)

    print('====================')

    return train_dataloader, eval_dataloader