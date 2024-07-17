import math
import os
from pathlib import Path
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import (
    AdamW, 
    get_scheduler, 
    is_wandb_available,
)
from tqdm import tqdm
from utils.math import multiply_grads, get_grad_norm

from dotenv import load_dotenv

dotenv_path = Path('../..', ".env")
load_dotenv(dotenv_path=dotenv_path)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = get_logger(__name__)

class Trainer:

    def __init__(self, 
                 args,
                 model, 
                 train_dataloader,
                 eval_dataloader,
                 accelerator,
                 api,
                 repo_id):

        self.model = model

        self.args = args

        self.api = api
        self.repo_id = repo_id

        self.accelerator = accelerator

        # Optimizer
        self.optimizer = AdamW(
            list(model.parameters()),
            lr=args.learning_rate,
            betas=[args.adam_beta1, args.adam_beta2],
            eps=args.adam_epsilon,
        )

        if args.env == 'server':
            # Prepare everything with our `accelerator`.
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
                model, self.optimizer, train_dataloader, eval_dataloader
            )
            # Gradient checkpointing
            self.model.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})


        if args.env == 'local':
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = model, self.optimizer, train_dataloader, eval_dataloader 


        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)

        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # 5. Train
        self.total_batch_size = args.per_device_train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps


    def train(self):
        
        args = self.args

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        completed_steps = 0
        starting_epoch = 0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        print("Starting training for {} epochs".format(args.num_train_epochs))
        for epoch in range(starting_epoch, args.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):

                # compute num of losses

                num_losses = batch["mask_time_indices"].sum()
                sub_attention_mask = batch.pop("sub_attention_mask", None)
                sub_attention_mask = (
                    sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
                )
                percent_masked = num_losses / sub_attention_mask.sum()

                # forward
                outputs = self.model(**batch)

                # divide loss by gradient accumulation steps since gradients
                # are accumulated for multiple backward passes in PyTorch
                loss = outputs.loss / args.gradient_accumulation_steps
                self.accelerator.backward(loss)

                # make sure that `num_losses` is summed for distributed training
                # and average gradients over losses of all devices
                if self.accelerator.state.num_processes > 1:
                    num_losses = self.accelerator.gather_for_metrics(num_losses).sum()
                    gradient_multiplier = self.accelerator.state.num_processes / num_losses
                    multiply_grads(self.model.module.parameters(), gradient_multiplier)
                else:
                    multiply_grads(self.model.parameters(), 1 / num_losses)

                # update step
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    # compute grad norm for monitoring
                    scale = (
                        self.accelerator.scaler._scale.item()
                        if hasattr(self.accelerator, "scaler") and self.accelerator.scaler is not None
                        else 1
                    )
                    if self.accelerator.state.num_processes > 1:
                        grad_norm = get_grad_norm(self.model.module.parameters(), scale)
                    else:
                        grad_norm = get_grad_norm(self.model.parameters(), scale)

                    # update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if not self.accelerator.optimizer_step_was_skipped:
                        self.lr_scheduler.step()
                    elif self.accelerator.is_local_main_process:
                        progress_bar.write(
                            f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                        )

                    # update gumbel temperature
                    gumbel_temperature = max(
                        args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
                        args.min_gumbel_temperature,
                    )
                    if hasattr(self.model, "module"):
                        self.model.module.set_gumbel_temperature(gumbel_temperature)
                    else:
                        self.model.set_gumbel_temperature(gumbel_temperature)

                    progress_bar.update(1)
                    completed_steps += 1

                # 6. Log all results
                if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                    loss.detach()
                    outputs.contrastive_loss.detach()
                    outputs.diversity_loss.detach()

                    if self.accelerator.state.num_processes > 1:
                        loss = self.accelerator.gather_for_metrics(loss).sum()
                        outputs.contrastive_loss = self.accelerator.gather_for_metrics(outputs.contrastive_loss).sum()
                        outputs.diversity_loss = self.accelerator.gather_for_metrics(outputs.diversity_loss).sum()
                        outputs.pose_diversity_loss = self.accelerator.gather_for_metrics(outputs.pose_diversity_loss).sum()
                        outputs.right_hand_diversity_loss = self.accelerator.gather_for_metrics(outputs.right_hand_diversity_loss).sum()
                        outputs.left_hand_diversity_loss = self.accelerator.gather_for_metrics(outputs.left_hand_diversity_loss).sum()
                        outputs.face_diversity_loss = self.accelerator.gather_for_metrics(outputs.face_diversity_loss).sum()
                        percent_masked = self.accelerator.gather_for_metrics(percent_masked).sum()

                    train_logs = {
                        "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                        "constrast_loss": outputs.contrastive_loss / num_losses,
                        "div_loss": outputs.diversity_loss / num_losses,
                        "%_mask_idx": percent_masked / self.accelerator.num_processes,
                        "ppl": outputs.codevector_perplexity,
                        "lr": torch.tensor(self.optimizer.param_groups[0]["lr"]),
                        "temp": torch.tensor(gumbel_temperature),
                        "grad_norm": torch.tensor(grad_norm),
                    }

                    if args.use_multi_cue:
                        train_logs["pose_diversity_loss"] = outputs.pose_diversity_loss / num_losses
                        train_logs["right_hand_diversity_loss"] = outputs.right_hand_diversity_loss / num_losses
                        train_logs["left_hand_diversity_loss"] = outputs.left_hand_diversity_loss / num_losses
                        train_logs["face_diversity_loss"] = outputs.face_diversity_loss / num_losses

                    log_str = ""
                    for k, v in train_logs.items():
                        log_str += "| {}: {:.3e}".format(k, v.item())

                    if self.accelerator.is_local_main_process:
                        progress_bar.write(log_str)
                        if is_wandb_available():
                            wandb.log(train_logs)

                # save model every `args.saving_steps` steps
                if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                    if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                        self.accelerator.wait_for_everyone()
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        unwrapped_model.save_pretrained(
                            args.output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
                        )

                    if (args.push_to_hub and epoch < args.num_train_epochs - 1) and self.accelerator.is_main_process:
                        self.api.upload_folder(
                            commit_message=f"Training in progress epoch {epoch}",
                            folder_path=args.output_dir,
                            repo_id=self.repo_id,
                            repo_type="model",
                            token=os.getenv("HUB_TOKEN"),
                        )

                # if completed steps > `args.max_train_steps` stop
                if completed_steps >= args.max_train_steps:
                    break

            # 7. Validate!
            self.model.eval()

            # # init logs
            val_logs = {
                "val_loss": 0,
                "val_contrastive_loss": 0,
                "val_diversity_loss": 0,
                "val_num_losses": 0,
            }

            if args.use_multi_cue:
                val_logs["val_pose_diversity_loss"] = 0
                val_logs["val_right_hand_diversity_loss"] = 0
                val_logs["val_left_hand_diversity_loss"] = 0
                val_logs["val_face_diversity_loss"] = 0

            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    batch.pop("sub_attention_mask", None)
                    outputs = self.model(**batch)

                val_logs["val_loss"] += outputs.loss
                val_logs["val_contrastive_loss"] += outputs.contrastive_loss
                val_logs["val_diversity_loss"] += outputs.diversity_loss
                val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

                if args.use_multi_cue:
                    val_logs["val_pose_diversity_loss"] += outputs.pose_diversity_loss
                    val_logs["val_right_hand_diversity_loss"] += outputs.right_hand_diversity_loss
                    val_logs["val_left_hand_diversity_loss"] += outputs.left_hand_diversity_loss
                    val_logs["val_face_diversity_loss"] += outputs.face_diversity_loss

            # sum over devices in multi-processing
            if self.accelerator.num_processes > 1:
                val_logs = {k: self.accelerator.gather_for_metrics(v).sum() for k, v in val_logs.items()}

            val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

            log_str = ""
            for k, v in val_logs.items():
                log_str += "| {}: {:.3e}".format(k, v.item())

            if self.accelerator.is_local_main_process:
                progress_bar.write(log_str)
                if is_wandb_available():
                    wandb.log(val_logs)

            if args.output_dir is not None:
                print("Waiting for everyone to save model")
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
                )
                print("Waiting for everyone to save model")
                if self.accelerator.is_main_process:
                    print("Saving model")
                    if args.push_to_hub:
                        self.api.upload_folder(
                            commit_message="End of training",
                            folder_path=args.output_dir,
                            repo_id=self.repo_id,
                            repo_type="model",
                            token=os.getenv("HUB_TOKEN"),
                        )