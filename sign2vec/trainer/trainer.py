import torch
import wandb
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Tokenizer, Wav2Vec2Config
from datasets import load_dataset

# Load the tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base")

# Load the Wav2Vec2 model for pretraining
model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

# Load a dataset for pretraining (use a dataset that provides raw audio files)
dataset = load_dataset("common_voice", "en", split="train")

# Define a data collator for processing the data
def data_collator(batch):
    input_values = tokenizer(batch["audio"]["array"], return_tensors="pt", padding="longest").input_values
    return {"input_values": input_values}

# Define a training loop
def train(model, dataset, tokenizer, epochs=1, batch_size=8, learning_rate=5e-5):
    # Set up data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            input_values = batch["input_values"]

            # Forward pass
            outputs = model(input_values, return_loss=True)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the pretrained model
    model.save_pretrained("wav2vec2-pretrained")

# Train the model
train(model, dataset, tokenizer)



class Wav2Vec2Pretraining():

    def __init__(self,
                 model,
                 tokenizer,
                 loss,
                 optimizer) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.loss = loss
        self.optimizer = optimizer

        # initilaze wandb for logging
        wandb.init(project="wav2vec2-pretraining")

    def train(self, dataset, epochs=1, batch_size=8, learning_rate=5e-5):
        # Set up data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for batch in data_loader:
                self.optimizer.zero_grad()
                input_values = batch["input_values"]

                # Forward pass
                outputs = self.model(input_values, return_loss=True)
                loss = outputs.loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()


                # Log the loss to wandb
                wandb.log({
                    "training-loss": loss.item()
                })

                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Save the pretrained model
        self.model.save_pretrained("wav2vec2-pretrained")