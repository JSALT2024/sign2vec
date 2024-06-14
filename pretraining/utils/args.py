import os
import argparse
from transformers import SchedulerType

def parse_args():

    parser = argparse.ArgumentParser(description="Pre-Training a 🤗 Sign2Vec2 model on unlabeled sign language data")

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--preprocessing_only",
        action="store_true",
        help="Only run the preprocessing script to be cached for future use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=1,
        help="Percentage of training data that should be used for validation if no validation is present in dataset.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--audio_column_name",
        type=str,
        default="audio",
        help="Column in the dataset that contains speech file path. Defaults to 'audio'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_cache_file_name",
        type=str,
        default=None,
        help="Path to the train cached file name",
    )
    parser.add_argument(
        "--validation_cache_file_name",
        type=str,
        default=None,
        help="Path to the validation cached file name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=5.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=3.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help=(
            "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the"
            " use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--mask_time_prob",
        type=float,
        default=None,
        help=(
            "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the"
            " contrastive task. If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--mask_time_length",
        type=int,
        default=None,
        help=(
            "Length of each vector mask span to mask along the time axis in the contrastive task."
            " If omitted, will pull value from model config."
        ),
    )


    # sign2vec specific arguments

    parser.add_argument(
        "--use_face",
        action="store_true",
        help="[sign2vec] Whether to use face features in the model",
    )

    parser.add_argument(
        "--use_hands",
        action="store_true",
        help="[sign2vec] Whether to use hand features in the model",
    )

    parser.add_argument(
        "--use_pose",
        action="store_true",
        help="[sign2vec] Whether to use pose features in the model",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="[sign2vec] Stride for the feature extraction",
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help="[sign2vec] Path to the training data",
    )

    parser.add_argument(
        "--train_info_path",
        type=str,
        default='info.json',
        help="[sign2vec] Path to the training info",
    )

    parser.add_argument(
        "--validation_data_path",
        type=str,
        default=None,
        help="[sign2vec] Path to the validation data",
    )

    parser.add_argument(
        "--validation_info_path",
        type=str,
        default=None,
        help="[sign2vec] Path to the validation info",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="[sign2vec] Frames per second in the video",
    )
    

    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args