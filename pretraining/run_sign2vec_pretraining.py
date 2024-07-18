"""Pre-Training a 🤗 Wav2Vec2 model on unlabeled audio data"""

import json
from accelerate.logging import get_logger
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from transformers.utils import send_example_telemetry

from utils.train import Trainer
from utils.args import parse_args
from sign2vec.config import Sign2VecConfig
from utils.accelerator import initialize_accelerator
from sign2vec.modeling_sign2vec import (
    Sign2VecForPreTraining, 
    MultiCueSign2VecForPreTraining
)


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
    # config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    config = Sign2VecConfig(**json.load(open(args.config_name)))


    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and"
            " ``config.feat_extract_norm='layer'"
        )

    # initialize random model
    model = Sign2VecForPreTraining(config) if not args.use_multi_cue else MultiCueSign2VecForPreTraining(config)
    if config.use_multi_cue:
        print("Using MultiCueSign2VecForPreTraining")
        print(model)
    # Import dataset and tokenizer
    from utils.dataset import prepare_dataloader
    train_dataloader, validation_dataloader = prepare_dataloader(args, config, model, accelerator)

    # Initialize training

    trainer = Trainer(args,
                      model, 
                      train_dataloader,
                      validation_dataloader,
                      accelerator,
                      api,
                      repo_id)
    # Training
    trainer.train()

if __name__ == "__main__":
    main()
