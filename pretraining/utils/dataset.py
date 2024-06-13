import datasets
from transformers import Wav2Vec2FeatureExtractor
from utils.collator import DataCollatorForWav2Vec2Pretraining
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data.dataloader import DataLoader

def prepare_dataset(args, config, model, accelerator):

    # 1. Download and create train, validation dataset
    # We load all dataset configuration and datset split pairs passed in
    # ``args.dataset_config_names`` and ``args.dataset_split_names``
    datasets_splits = []
    for dataset_config_name, train_split_name in zip(args.dataset_config_names, args.dataset_split_names):
        # load dataset
        dataset_split = load_dataset(
            args.dataset_name, # NOTE: GET THIS FROM ARGS -> should be 'jsalt/bobsl_v1'
            dataset_config_name, 
            split=train_split_name,
            cache_dir=args.cache_dir,
        )
        datasets_splits.append(dataset_split)


    # Next, we concatenate all configurations and splits into a single training dataset
    raw_datasets = DatasetDict()
    if len(datasets_splits) > 1:
        raw_datasets["train"] = concatenate_datasets(datasets_splits).shuffle(seed=args.seed)
    else:
        raw_datasets["train"] = datasets_splits[0]

    # Take ``args.validation_split_percentage`` from the training dataset for the validation_split_percentage
    num_validation_samples = raw_datasets["train"].num_rows * args.validation_split_percentage // 100

    if num_validation_samples == 0:
        raise ValueError(
            "`args.validation_split_percentage` is less than a single sample "
            f"for {len(raw_datasets['train'])} training samples. Increase "
            "`args.num_validation_split_percentage`. "
        )

    raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
    raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))

    # 2. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)

    # make sure that dataset decodes audio with correct sampling rate
    raw_datasets = raw_datasets.cast_column(
        args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    # only normalized-inputs-training is supported
    if not feature_extractor.do_normalize:
        raise ValueError(
            "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
        )

    # set max & min audio length in number of samples
    max_length = int(args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_length = int(args.min_duration_in_seconds * feature_extractor.sampling_rate)

    def prepare_dataset(batch):
        # TODO: Change this for sign2vec
        sample = batch[args.audio_column_name]

        inputs = feature_extractor(
            sample["array"], max_length=max_length, truncation=True
        )
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(inputs.input_values[0])

        return batch

    # load via mapped files via path
    cache_file_names = None
    if args.train_cache_file_name is not None:
        cache_file_names = {"train": args.train_cache_file_name, "validation": args.validation_cache_file_name}

    # load audio files into numpy arrays
    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            cache_file_names=cache_file_names,
        )

        if min_length > 0.0:
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: x > min_length,
                num_proc=args.preprocessing_num_workers,
                input_columns=["input_length"],
            )

        vectorized_datasets = vectorized_datasets.remove_columns("input_length")

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

    return train_dataloader, eval_dataloader, vectorized_datasets