import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path so that we can import the modules
sys.path.append(os.path.dirname('../'))

from torch import nn
from transformers import T5ForConditionalGeneration
from sign2vec.modeling_sign2vec import Sign2VecModel



class T5ForSignLanguageTranslation(nn.Module):

	def __init__(self, 
				 sign2vec_model: Sign2VecModel = None,
				 t5_model: T5ForConditionalGeneration = None,
				 sign2vec_embed_dim: int = 1024,
				 t5_embed_dim: int = 512,
				 dropout: float = 0.1,
				 *args, 
				 **kwargs,
				 ) -> None:
		super().__init__(*args, **kwargs)

		# Load the sign2vec model
		self.sign2vec = sign2vec_model

		for param in self.sign2vec.parameters():
			param.requires_grad = False

		# Linear layer to project the sign2vec embeddings to the T5 embeddings
		self.linear = nn.Linear(sign2vec_embed_dim, t5_embed_dim)
		# self.layer_norm = nn.LayerNorm(t5_embed_dim)
		self.dropout = nn.Dropout(dropout)

		# Load the T5 model
		self.t5 = t5_model
		self.t5.encoder.embed_tokens = nn.Linear(512, self.t5.config.d_model)

		# for param in self.t5.parameters():
		# 	param.requires_grad = False

		# self.t5.lm_head.requires_grad = True



	def generate(self, input_values, **kwargs):
		
		sign2vec_out = self.sign2vec(input_values)
		out_proj = self.linear(sign2vec_out.last_hidden_state)
		# out_proj = self.layer_norm(out_proj)
		out_proj = self.dropout(out_proj)

		outputs = self.t5.generate(
			inputs_embeds = out_proj,
			**kwargs
		)

		return outputs

	def forward(self, input_values, decoder_input_ids, **kwargs):

		sign2vec_out = self.sign2vec(input_values)
		out_proj = self.linear(sign2vec_out.last_hidden_state)
		out_proj = self.dropout(out_proj)

		outputs = self.t5(
			labels=decoder_input_ids,
			inputs_embeds = out_proj,
		)

		return outputs
	

class T5BaseForSignLanguageTranslation(nn.Module):

	def __init__(self, 
				 t5_model: T5ForConditionalGeneration = None,
				 input_dim: int = 1024,
				 t5_embed_dim: int = 512,
				 dropout: float = 0.1,
				 *args, 
				 **kwargs,
				 ) -> None:
		super().__init__(*args, **kwargs)

		self.linear = nn.Linear(input_dim, t5_embed_dim)
		self.dropout = nn.Dropout(dropout)

		self.t5 = t5_model
		self.t5.encoder.embed_tokens = nn.Linear(input_dim, self.t5.config.d_model)
	
		# for param in self.t5.parameters():
		# 	param.requires_grad = False

		# self.t5.lm_head.requires_grad = True

	def generate(self, input_values, **kwargs):

		out_proj = self.linear(input_values.transpose(1, 2))
		out_proj = self.dropout(out_proj)

		outputs = self.t5.generate(
			inputs_embeds = out_proj,
			**kwargs
		)

		return outputs

	def forward(self, input_values, decoder_input_ids, **kwargs):

		out_proj = self.linear(input_values.transpose(1, 2))
		out_proj = self.dropout(out_proj)

		outputs = self.t5(
			labels=decoder_input_ids,
			inputs_embeds = out_proj,
		)

		return outputs
	