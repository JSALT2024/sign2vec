"""Pre-Training a 🤗 Wav2Vec2 model on unlabeled audio data"""


from accelerate.logging import get_logger
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
)
from transformers.utils import send_example_telemetry

from utils.train import Trainer
from utils.args import parse_args
from utils.accelerator import initialize_accelerator

logger = get_logger(__name__)

def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_wav2vec2_pretraining_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator, api, repo_id = initialize_accelerator(args)

    # Initialize random model before training
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)

    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and"
            " ``config.feat_extract_norm='layer'"
        )

    # initialize random model
    model = Wav2Vec2ForPreTraining(config)

    # Import dataset and tokenizer
    from utils.dataset import prepare_dataset
    train_dataset, validation_dataset, vectorized_datasets = prepare_dataset(args, config, model, accelerator)

    # Initialize training
    trainer = Trainer(args, model, train_dataset, validation_dataset, vectorized_datasets, accelerator, api, repo_id)

    # Training
    trainer.train()

if __name__ == "__main__":
    main()
