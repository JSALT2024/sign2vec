# How to

In this document, we will explain the subjects following

- How2Sign T5 Training
- YoutubeASL T5 Pre-Training

Before you start, you need to setup your environment:

### a. Dataset Format

#### a.1 How2Sign (Single-h5 format)

* For How2Sign, it is enough to have only the h5 files:

```
h2s
|--- H2S_train.h5
|--- H2S_val.h5
|--- H2S_test.h5
```

In these files, if your modality is `pose`, each h5 item should look like as below:

```json
{
    "clip_id": {
        "joints": {
            "pose_landmarks": np.ndarray(sequence_length, num_joints, 4)
            "left_hand_landmarks": np.ndarray(sequence_length, num_joints, 4)
            "right_hand_landmarks": np.ndarray(sequence_length, num_joints, 4)
            "face_landmarks": np.ndarray(sequence_length, num_joints, 4)
        }
        "sentence": "Hi!"
    }
}
```

if your modality is a latent representation, it is enough have it like this.

```json
{
    "clip_id": {
        "features": np.ndarray(sequence_length, embedding_dim)
        "sentence": "Hi!"
    }
}
```

#### a.2 YoutubeASL (Multi-h5 format)

* For YoutubeASL, you need to generate an additional `csv` file for each modality and put them in the same folder with h5 files. The csv files should include: h5 file_name, original clip_id, file index of clip. We have provided the our split in the `data` folder.

```
yasl
  |--- yasl_pose_0.h5
  |--- yasl_pose_1.h5
  |--- yasl_pose_2.h5
  |--- ......
  |--- train_dataset.csv
  |--- val_dataset.csv
```

### b. Adding custom modality

- You need to add custom dataset class to `sign2vec/dataset/dataset_name.py` file 

```python
class YoutubeASLForCustom(Dataset):

    def __init__(
        self,
        h5_fpath,
    ):
        self.h5_file = h5py.File(h5_fpath, "r")

    def __len__(self):
        return len(list(self.h5_file.keys())) 

    def __getitem__(self, idx):
        
        data = self.h5_file[list(self.h5_file.keys())[idx]]

        sign2vec = data["features"][()]
        sentence = data["sentence"][()].decode("utf-8")

        ## YOUR TRANSFORMATION HERE

        return sign2vec, sentence
```

* Then modify the dataset `DatasetNameForSLT` class for your custom dataset 

```python
class YoutubeASLForSLT(YoutubeASLForPose, YoutubeASLForSign2Vec, YoutubeASLForCustom):
    
    def __init__(...):

        ...

        YoutubeASLForCustom.__init__(self, self.h5_file_name, ...)


    def __getitem__(self, idx)
        if self.input_type == "custom":
            # Reinitialize the dataset if the h5 file is different
            if self.h5_file_name != h5_file:
                YoutubeASLForCustom.__init__(self, h5_file, self.max_instances)
            keypoints, sentence = YoutubeASLForSign2Vec.__getitem__(self, file_idx)
```

* Finally, use the modality as `custom` to ``run_t5_trainer.py`` argument

## T5 Training

We are using the default parameters coming from the YoutubeASL paper where the source is coming from <a href="https://github.com/google-research/t5x/blob/main/t5x/configs/runs/finetune.gin">TSX library from Google</a>. Although, we provide option for changing, we suggest not to change it for replication purposes.

The bash command should look like as below.

```bash
python3 sign2vec.train.run_t5_training --model_name=h2s-test \
                                       --model_id=google-t5/t5-base \
                                       --dataset_dir=path/to/h5 \
                                       --output_dir=path/to/output \
                                       --modality="your modality" \ # Currently supports "pose" and "sign2vec"
                                       --skip_frames \ # This is from YoutubeASL paper
                                       --max_token_length=128 \
                                       --max_sequence_length=250 \
                                       --max_training_steps=250 \
                                       --per_device_train_batch_size=16 \
                                       --gradient_accumulation_steps=8 \
                                       --embedding_dim=255 \ # This is for pose 

```

### Experimental Parameters:

| Dataset    | Modality | Embedding Dim | Learning Rate | Optimizer | Max Steps |
| ---------- | -------- | ------------- | ------------- | --------- | --------- |
| How2Sign   | Pose     | 255           | 0.001         | Adafactor | 20_000    |
| YoutubeASL | Pose     | 255           | 0.001         | Adafactor | 200_000   |




