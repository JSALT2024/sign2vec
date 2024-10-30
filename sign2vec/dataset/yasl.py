import os
import h5py
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from sign2vec.utils.normalization import normalize_local, normalize_global, normalize_yasl

POSE_LANDMARKS = [11, 12, 13, 14, 23, 24]

FACE_LANDMARKS = [
    0, 4, 13, 14, 17, 33, 37, 39, 46,
    52, 55, 61, 64, 81, 82, 93,
    133, 151, 152, 159, 172, 178, 181, 
    263, 269, 276, 282, 285, 291, 294,
    311, 323, 362, 386, 397,
    468, 473 
]


class YoutubeASLForPose(Dataset):

    def __init__(
        self,
        h5_file,
        transform = 'yasl',
        max_instances=None,
        zero_mean=False,
    ):
        self.transform = transform
        self.h5_file = h5py.File(h5_file, "r")
        self.max_instances = max_instances
        self.zero_mean = zero_mean

    def __len__(self):
        return len(list(self.h5_file.keys())) if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):
        
        clip_id = list(self.h5_file.keys())[idx]

        data = self.h5_file[clip_id]

        pose_landmarks = data["joints"]["pose_landmarks"][()]
        face_landmarks = data["joints"]["face_landmarks"][()]
        left_hand_landmarks = data["joints"]["left_hand_landmarks"][()]
        right_hand_landmarks = data["joints"]["right_hand_landmarks"][()]
        # sentence = data["sentence"][()].decode("utf-8")

        if self.transform:
            if self.transform == 'yasl':
                pose_landmarks = normalize_yasl(pose_landmarks)
                face_landmarks = normalize_yasl(face_landmarks)
                left_hand_landmarks = normalize_yasl(left_hand_landmarks)
                right_hand_landmarks = normalize_yasl(right_hand_landmarks)
            else:
                raise NotImplementedError(f'{self.transform} normalization is not implemented yet')

        # Select only the keypoints that are needed
        pose_landmarks = pose_landmarks[:, POSE_LANDMARKS, :]
        face_landmarks = face_landmarks[:, FACE_LANDMARKS, :]

        # Remove last 1 channel (visibility)
        pose_landmarks = pose_landmarks[:, :, :-1]
        face_landmarks = face_landmarks[:, :, :-1]
        left_hand_landmarks = left_hand_landmarks[:, :, :-1]
        right_hand_landmarks = right_hand_landmarks[:, :, :-1]

        # Convert keypoints to tensor
        pose_landmarks = torch.tensor(pose_landmarks, dtype=torch.float)
        face_landmarks = torch.tensor(face_landmarks, dtype=torch.float)
        left_hand_landmarks = torch.tensor(left_hand_landmarks, dtype=torch.float)
        right_hand_landmarks = torch.tensor(right_hand_landmarks, dtype=torch.float)

        # Concatenate all keypoints
        keypoints = torch.cat(
            (pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks),
            dim=1,
        )
        # Reduce the keypoints (T, N, C) -> (T, N*C)
        keypoints = keypoints.view(keypoints.size(0), -1) 
        # Check if keypoints are in the correct shape
        assert keypoints.shape[-1] == 255, "Key points are not in the correct shape"

        # Replace NaN values with 0
        torch.nan_to_num_(keypoints, nan=0.0)

        # Apply zero mean normalization to each keypoint dim
        if self.zero_mean:
            keypoints = (keypoints - keypoints.mean(dim=0)) / keypoints.std(dim=0)

        # Replace NaN, Inf values with 0
        torch.nan_to_num_(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

        return keypoints #, sentence


class YoutubeASLForSign2Vec(Dataset):

    def __init__(
        self,
        h5_fpath,
        max_instances=None,
    ):
        self.h5_file = h5py.File(h5_fpath, "r")
        self.max_instances = max_instances

    def __len__(self):
        return len(list(self.h5_file.keys())) if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):
        
        data = self.h5_file[list(self.h5_file.keys())[idx]]

        sign2vec = data["features"][()]
        sentence = data["sentence"][()].decode("utf-8")

        return sign2vec, sentence

class YoutubeASLForSLT(YoutubeASLForPose, YoutubeASLForSign2Vec):

    def __init__(
        self,
        h5_fpath,
        mode="train",
        input_type="pose",
        skip_frames=True,
        transform=[("pose_landmarks", "local"), ("face_landmarks", "local")],
        max_token_length=128,
        max_sequence_length=250,
        tokenizer="google-t5/t5-small",
        max_instances=None,
    ):

        self.mode = mode
        self.h5_fpath = h5_fpath
        self.csv_file = pd.read_csv(os.path.join(h5_fpath, f"yasl_{mode}.csv"))
        self.csv_file['h5_file'] = self.csv_file['h5_file'].apply(lambda x: os.path.join(h5_fpath, x))
        
        self.h5_file_name = self.csv_file.iloc[0].h5_file

        self.input_type = input_type

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames

        if self.input_type == "sign2vec" and skip_frames: raise ValueError("skip_frames should be False for `sign2vec` input")

        YoutubeASLForPose.__init__(self, self.h5_file_name, transform, max_instances)
        YoutubeASLForSign2Vec.__init__(self, self.h5_file_name, max_instances)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # Get the keypoints and the sentence

        h5_file = self.csv_file.iloc[idx].h5_file
        file_idx = self.csv_file.iloc[idx].file_idx
        sentence = self.csv_file.iloc[idx].sentence

        if self.input_type == "pose":
            # Reinitialize the dataset if the h5 file is different
            if self.h5_file_name != h5_file:
                YoutubeASLForPose.__init__(self, h5_file, self.transform, self.max_instances)
            keypoints = YoutubeASLForPose.__getitem__(self, file_idx)

        elif self.input_type == "sign2vec":
            # Reinitialize the dataset if the h5 file is different
            if self.h5_file_name != h5_file:
                YoutubeASLForSign2Vec.__init__(self, h5_file, self.max_instances)
            keypoints = YoutubeASLForSign2Vec.__getitem__(self, file_idx)

        self.h5_file_name = h5_file

        # Tokenize the sentence
        decoder_input_ids = self.tokenizer(
            sentence,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # Shift the token ids using tokenizer._shift_tokens_right
        # decoder_input_ids = self.tokenizer._shift_right(decoder_input_ids)

        # Skip frames for the keypoints
        if self.skip_frames: keypoints = keypoints[::2]
        # Trim the keypoints to the max sequence length
        keypoints = keypoints[: self.max_sequence_length]
        attention_mask = torch.ones(len(keypoints))

        return {
            "inputs": keypoints,
            "sentence": sentence,
            "labels": decoder_input_ids,
            "attention_mask": attention_mask,
        }


class YoutubeASLForSign2VecPretraining(YoutubeASLForPose):

    def __init__(
        self,
        h5_fpath,
        mode="train",
        transform=[("pose_landmarks", "local"), ("face_landmarks", "local")],
        skip_frames=False,
        max_sequence_length=None,
        add_factor=0.1,
        zero_mean=False,
    ):
        
        self.mode = mode
        print(f"Loading {self.mode} data")
        print(f"Loading data from {h5_fpath}")
        self.csv_file = pd.read_csv(os.path.join(h5_fpath, f'yasl_{self.mode}.csv'))
        self.csv_file['h5_file'] = self.csv_file['h5_file'].apply(lambda x: os.path.join(h5_fpath, x))

        self.h5_file_name = self.csv_file.iloc[0].h5_file

        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames
        self.add_factor = add_factor

        YoutubeASLForPose.__init__(self, self.h5_file_name, transform, zero_mean)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        h5_file, file_idx = self.csv_file.iloc[idx].h5_file, self.csv_file.iloc[idx].file_idx

        # Reinitialize the dataset if the h5 file is different
        if self.h5_file_name != h5_file:
            YoutubeASLForPose.__init__(self, h5_file, self.transform)
        
        self.h5_file_name = h5_file

        keypoints = YoutubeASLForPose.__getitem__(self, file_idx)

        if self.skip_frames: keypoints = keypoints[::2]
        if self.max_sequence_length: keypoints = keypoints[: self.max_sequence_length]
        keypoints = keypoints * self.add_factor

        return {
            "input_values": keypoints,
        }
