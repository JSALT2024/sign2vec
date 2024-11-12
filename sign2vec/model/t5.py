import torch
from torch import nn
from sign2vec.utils.translation import collate_fn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

# Subclassing the T5 model
class T5ModelForSLT(nn.Module):
    def __init__(self, model_name_or_path, config):
        super(T5ModelForSLT, self).__init__()

        # Define a custom linear layer to apply to the input embeddings
        self.custom_linear = nn.Linear(config.pose_dim, config.d_model)
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        
        self.device = self.base_model.device

    def generate(self, **kwargs):

        kwargs["inputs_embeds"] = self.custom_linear(kwargs["sign_inputs"])
        kwargs.pop("sign_inputs")

        kwargs["input_ids"] = None

        return self.base_model.generate(**kwargs)
    
    # Override the forward method to modify input embeddings
    def forward(
        self,
        sign_inputs=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # cache_position=None,
    ):
        
        # Apply custom linear layer to the input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.custom_linear(sign_inputs)

        # Pass modified embeddings to the original T5 forward method
        return self.base_model.forward(
            input_ids=None,  # We use inputs_embeds instead of input_ids
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position=cache_position,
        )

if __name__ == "__main__":

    import os
    from sign2vec.dataset.how2sign import How2SignForSLT

    MODEL_ID = "t5-small"
    H2S_DIR = '/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign'

    # Initialize tokenizer and config
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    config = T5Config.from_pretrained(MODEL_ID)
    config.pose_dim = 255  # Dimension of the pose embeddings

    # Initialize the custom model
    model = T5ModelForSLT(config)

    # Import How2SignForSLT dataset
    h2s_test = How2SignForSLT(
        h5_fpath=os.path.join(H2S_DIR, 'Mediapipe','H2S_test.h5'),   
        transform=None,
        max_token_length=128,
        max_sequence_length=250,
        skip_frames=True,
        tokenizer=MODEL_ID
    )

    # Initialize DataLoader
    test_loader = torch.utils.data.DataLoader(
        h2s_test,
        batch_size=5,
        shuffle=False,
        collate_fn=collate_fn,
    )

    inputs = next(iter(test_loader))

    print('Model custom linear layer:', model.custom_linear.weight.shape)
    print('Model custom linear layer:', model.custom_linear.weight)

    for key, _input in inputs.items():
        print(key,_input.shape, _input)

    # Forward pass through the model
    outputs = model(**inputs, labels=inputs["decoder_input_ids"])

    # Get the loss and logits
    loss = outputs.loss
    logits = outputs.logits

    # Decode batch of logits
    for i in range(logits.shape[0]):
        predicted_sentences = tokenizer.decode(torch.argmax(logits[i], dim=1))
        print(f"Predicted sentence: {predicted_sentences}")

    print(f"Loss: {loss.item()}")

