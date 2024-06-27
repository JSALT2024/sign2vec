import os
from pathlib import Path

import datasets
import transformers
from huggingface_hub import HfApi
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import is_wandb_available, set_seed
from accelerate.utils import DistributedDataParallelKwargs

from dotenv import load_dotenv

dotenv_path = Path('../', ".env")
load_dotenv(dotenv_path=dotenv_path)

logger = get_logger(__name__)

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

def initialize_accelerator(args):

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

        # set up weights and biases if available
        if is_wandb_available():
            import wandb

            wandb.init(
                project=args.output_dir.split("/")[-1]
            )
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    api = None
    repo_id = None
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub and not args.preprocessing_only:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=os.getenv("HUB_TOKEN")).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")

                gitignore.write("wandb/\n")
                gitignore.write(".env\n")

        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    return accelerator, api, repo_id
