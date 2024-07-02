from torch import nn
from transformers import T5ForConditionalGeneration

# 2. Define Custom T5 Model with Linear Layer
class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.custom_linear = nn.Linear(162, config.d_model)  # Assuming input size is 128, adjust as needed
    
    def forward(self, continuous_input=None, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        if continuous_input is not None:
            continuous_input_ = self.custom_linear(continuous_input)
            encoder_outputs = self.encoder(inputs_embeds=continuous_input_, attention_mask=attention_mask)
        else:
            raise ValueError("continuous_input cannot be None")

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],  # encoder_outputs is a tuple, we need the first element
            encoder_attention_mask=attention_mask,
            use_cache=False,
        )

        sequence_output = decoder_outputs[0]
        logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, logits)
