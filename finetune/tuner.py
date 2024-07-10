import torch
import pytorch_lightning as pl
from sign2vec.dataset.how2sign_hf5 import How2SignDatasetForFinetuning
from sign2vec.models.finetune import T5BaseForSignLanguageTranslation
from transformers import ( 
  T5Tokenizer, 
  Adafactor,
  get_linear_schedule_with_warmup
)

import logging
from torch.utils.data import DataLoader
from finetune.collator import DataCollatorForSign2VecFinetuning

class T5FineTuner(pl.LightningModule):
  def __init__(self, params):
    super(T5FineTuner, self).__init__()
    self.hparam = params
    
    self.model = T5BaseForSignLanguageTranslation( 
      model_id=self.hparam.t5_model_path_or_name, embed_size=self.hparam.t5_embed_dim
    )

    for param in self.model.model.named_parameters():
        if self.hparam.freeze_encoder:
            if 'encoder' in param[0]:
                print('Freezing param:', param[0])
                param[1].requires_grad = False

        if self.hparam.freeze_decoder:
            if 'decoder' in param[0]:
                print('Freezing param:', param[0])
                param[1].requires_grad = False

    self.tokenizer = T5Tokenizer.from_pretrained(self.hparam.tokenizer_name_or_path)
  
  def is_logger(self):
    return self.trainer.proc_rank <= 0
  
  def forward(
      self, input_values, decoder_input_ids
  ):
    return self.model(
        input_values, decoder_input_ids
    )

  def _step(self, batch):
    outputs = self(**batch)
    return outputs.loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)
    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
#   def on_train_epoch_end(self, outputs):
#     avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
#     tensorboard_logs = {"avg_train_loss": avg_train_loss}
#     return {
#       "avg_train_loss": avg_train_loss, 
#       "log": tensorboard_logs, 
#       'progress_bar': tensorboard_logs
#     }

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {
      "val_loss": loss
    }
  
#   def on_validation_epoch_end(self, outputs):
#     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#     tensorboard_logs = {"val_loss": avg_loss}
#     return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    # 5. Define the optimizer and scheduler
    optimizer = Adafactor(
        model.parameters(),
        lr=self.hparam.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=self.hparam.weight_decay,
        relative_step=False,
        scale_parameter=False,
    )
    
    self.opt = optimizer
    return [optimizer]

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    # update params
    optimizer.step(closure=optimizer_closure)

    # manually warm up lr without a scheduler
    if self.trainer.global_step < 500:
        lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * self.hparam.learning_rate

    optimizer.zero_grad()
  
  def get_tqdm_dict(self):

    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = How2SignDatasetForFinetuning(
        dataset=self.hparam.train_dataset_path,
        data_dir=self.hparam.data_path,
        max_length=self.hparam.max_frames,
    )

    data_collator = DataCollatorForSign2VecFinetuning(
        model=None,
        feature_extractor=None,
        shift_right=False,
        pad_to_multiple_of=self.hparam.pad_to_multiple_of,
        mask_time_prob=self.hparam.mask_time_prob,
        mask_time_length=self.hparam.mask_time_length,
        tokenizer=self.tokenizer,
        max_frame_length=self.hparam.max_frames,
        max_text_length=self.hparam.max_sequence_length,
        skip_frames=2,
    )

    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=self.hparam.batch_size,
    )

    t_total = (
        (len(dataloader.dataset) // (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
        // self.hparam.gradient_accumulation_steps
        * float(self.hparam.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    train_dataset = How2SignDatasetForFinetuning(
        dataset=self.hparam.train_dataset_path,
        data_dir=self.hparam.data_path,
        max_length=self.hparam.max_frames,
    )

    data_collator = DataCollatorForSign2VecFinetuning(
        model=None,
        feature_extractor=None,
        shift_right=False,
        pad_to_multiple_of=self.hparam.pad_to_multiple_of,
        mask_time_prob=self.hparam.mask_time_prob,
        mask_time_length=self.hparam.mask_time_length,
        tokenizer=self.tokenizer,
        max_frame_length=self.hparam.max_frames,
        max_text_length=self.hparam.max_sequence_length,
        skip_frames=2,
    )

    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=self.hparam.batch_size,
    )
    return dataloader
     