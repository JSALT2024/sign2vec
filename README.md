# sign2vec


The implementation of wav2vec2.0 for SLR

## Usage

```bash
cd models
git clone git@github.com:JSALT2024/sign2vec.git
git checkout pretraining
git pull
cd ../..
````

```python
import torch
from sign2vec.modeling_sign2vec import Sign2VecModel
from sign2vec.feature_extraction_sign2vec import Sign2VecFeatureExtractor

model = Sign2VecModel.from_pretrained(
    'karahansahin/sign2vec-yasl-sc-sc-80-4-d1-decay', token=env.HF_TOKEN
)
feature_extractor = Sign2VecFeatureExtractor()

inputs = feature_extractor(
    sample_pose
)

features = inputs["input_values"][0]
features = torch.tensor(features).float()
features = features.transpose(1, 2)
out = sign2vec_model(
    features
)
sign2vec_features = out.last_hidden_state.detach().cpu().numpy()[0]
print(sign2vec_features)
#tensor([[[-0.1184, -0.1246, -0.1215,  ..., -0.1805, -0.1769, -0.1857],
#         [-0.9325, -0.9161, -0.9153,  ..., -1.0540, -1.0580, -1.0520],
#         [ 0.0312,  0.0257,  0.0299,  ..., -0.0291, -0.0258, -0.0367],
#         ...,
#         [-0.6116, -0.5978, -0.5964,  ..., -0.6146, -0.6115, -0.6030],
#         [ 0.4678,  0.4096,  0.4076,  ...,  0.4546,  0.4515,  0.4700],
#         [-0.6329, -0.6505, -0.6490,  ..., -0.6549, -0.6518, -0.6233]]])
```