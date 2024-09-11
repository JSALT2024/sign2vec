import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from sign2vec.utils.config import Sign2VecConfig
from sign2vec.utils.feature_extractor import Sign2VecFeatureEncoder
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2Encoder,
    Wav2Vec2Adapter,
    Wav2Vec2ForPreTraining,
    Wav2Vec2FeatureProjection,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2GumbelVectorQuantizer,
    WAV_2_VEC_2_START_DOCSTRING,
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _EXPECTED_OUTPUT_SHAPE,
)
from transformers.utils import (
    add_start_docstrings,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import ModelOutput

from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class Sign2VecBaseModelOutput(ModelOutput):
    """
    Base class for models that have been trained with the Wav2Vec2 loss objective.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    quantizations: Optional[Tuple[torch.FloatTensor]] = None
    quantized_features: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    "The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Sign2VecModel(Wav2Vec2Model):
    def __init__(self, config: Sign2VecConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Sign2VecFeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()


    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Sign2VecBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="pose",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Sign2VecBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Sign2VecBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Sign2VecForPreTraining(Wav2Vec2ForPreTraining):
    def __init__(self, config: Sign2VecConfig):
        super().__init__(config)
        self.wav2vec2 = Sign2VecModel(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)
        if config.is_multicue:
            self.pose_quantizer = Wav2Vec2GumbelVectorQuantizer(config)
            self.left_hand_quantizer = Wav2Vec2GumbelVectorQuantizer(config)
            self.right_hand_quantizer = Wav2Vec2GumbelVectorQuantizer(config)
            self.face_quantizer = Wav2Vec2GumbelVectorQuantizer(config)
            # if multicue, add additional quantizer
            raise NotImplementedError("Multicue not implemented yet")

        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature
        if self.config.is_multicue:
            self.pose_quantizer.temperature = temperature
            self.left_hand_quantizer.temperature = temperature
            self.right_hand_quantizer.temperature = temperature
            self.face_quantizer.temperature = temperature
            # if multicue, set temperature for all quantizers
            raise NotImplementedError("Multicue not implemented yet")