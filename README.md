# sign2vec

The implementation of `wav2vec2.0` for SLR

* copy how2sign from Royal

```
scp -J karahan@193.140.195.142 karahan@193.140.195.17:/ssd1/karahan/How2Sign/H2S_train.h5 /home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe/

scp -J karahan@193.140.195.142 karahan@193.140.195.17:/ssd1/karahan/How2Sign/H2S_val.h5 /home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe/

scp -J karahan@193.140.195.142 karahan@193.140.195.17:/ssd1/karahan/How2Sign/H2S_test.h5 /home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe/
```

## Usage


### 1. Sign2Vec Pretraining

```

```

### 2. YASL T5 Pretraining

```bash
python3 -m sign2vec.train.run_training_for_translation --experimental_config='/experimental/configs/yasl_t5.yaml'
```

### 3. How2Sign Finetuning

```bash
python3 -m sign2vec.train.run_training_for_translation --experimental_config='/experimental/configs/h2s_t5.yaml'
```
