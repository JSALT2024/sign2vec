import os
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from sign2vec.utils.normalization import normalize_local, normalize_global

POSE_LANDMARKS = [11, 12, 13, 14, 23, 24]

FACE_LANDMARKS = [
    0, 4, 13, 14, 17, 33, 37, 39, 46,
    52, 55, 61, 64, 81, 82, 93,
    133, 151, 152, 159, 172, 178, 181, 
    263, 269, 276, 282, 285, 291, 294,
    311, 323, 362, 386, 397,
    468, 473 
]


class How2SignForPose(Dataset):

    def __init__(
        self,
        h5_fpath,
        transform=[("pose_landmarks", "local"), ("face_landmarks", "local")],
        max_instances=None,
    ):
        self.transform = transform
        self.h5file = h5py.File(h5_fpath, "r")
        self.max_instances = max_instances

    def __len__(self):
        return len(list(self.h5file.keys())) if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):

        data = self.h5file[list(self.h5file.keys())[idx]]

        pose_landmarks = data["joints"]["pose_landmarks"][()]
        face_landmarks = data["joints"]["face_landmarks"][()]
        left_hand_landmarks = data["joints"]["left_hand_landmarks"][()]
        right_hand_landmarks = data["joints"]["right_hand_landmarks"][()]
        sentence = data["sentence"][()].decode("utf-8")

        if self.transform:
            for norm in self.transform:
                if norm[0] == "pose_landmarks":
                    if norm[1] == "local":
                        pose_landmarks = self.normalize_local(pose_landmarks, "pose")
                    elif norm[1] == "global":
                        pose_landmarks = self.normalize_global(pose_landmarks, "pose")
                    else:
                        raise ValueError("Unknown normalization method")
                elif norm[0] == "face_landmarks":
                    if norm[1] == "local":
                        face_landmarks = self.normalize_local(face_landmarks, "face")
                    elif norm[1] == "global":
                        face_landmarks = self.normalize_global(face_landmarks, "face")
                    else:
                        raise ValueError("Unknown normalization method")
                elif norm[0] == "left_hand_landmarks":
                    if norm[1] == "local":
                        left_hand_landmarks = self.normalize_local(
                            left_hand_landmarks, "hand"
                        )
                    elif norm[1] == "global":
                        left_hand_landmarks = self.normalize_global(
                            left_hand_landmarks, "hand"
                        )
                    else:
                        raise ValueError("Unknown normalization method")
                elif norm[0] == "right_hand_landmarks":
                    if norm[1] == "local":
                        right_hand_landmarks = self.normalize_local(
                            right_hand_landmarks, "hand"
                        )
                    elif norm[1] == "global":
                        right_hand_landmarks = self.normalize_global(
                            right_hand_landmarks, "hand"
                        )
                    else:
                        raise ValueError("Unknown normalization method")
                else:
                    raise ValueError("Unknown keypoint type")

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

        return keypoints, sentence


class How2SignForSign2Vec(Dataset):

    def __init__(
        self,
        h5_fpath,
        max_instances=None,
    ):
        self.h5file = h5py.File(h5_fpath, "r")
        self.max_instances = max_instances

    def __len__(self):
        return len(list(self.h5file.keys())) if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):
        
        data = self.h5file[list(self.h5file.keys())[idx]]

        sign2vec = data["features"][()]
        sentence = data["sentence"][()].decode("utf-8")

        return sign2vec, sentence

class How2SignForSLT(How2SignForPose, How2SignForSign2Vec):

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
        h5_fpath = os.path.join(h5_fpath, f"H2S_{mode}.h5")
        
        self.input_type = input_type

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames

        if self.input_type == "sign2vec" and skip_frames: raise ValueError("skip_frames should be False for `sign2vec` input")

        How2SignForPose.__init__(self, h5_fpath, transform, max_instances)
        How2SignForSign2Vec.__init__(self, h5_fpath, max_instances)

    def __getitem__(self, idx):
        # Get the keypoints and the sentence
        if self.input_type == "pose":
            keypoints, sentence = How2SignForPose.__getitem__(self, idx)
        elif self.input_type == "sign2vec":
            keypoints, sentence = How2SignForSign2Vec.__getitem__(self, idx)

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


class How2SignForSign2VecPretraining(How2SignForPose):

    def __init__(
        self,
        h5_fpath,
        mode="train",
        transform=[("pose_landmarks", "local"), ("face_landmarks", "local")],
        skip_frames=False,
        max_sequence_length=None,
    ):
        
        self.mode = mode
        h5_fpath = os.path.join(h5_fpath, f"H2S_{mode}.h5")

        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames

        super(How2SignForSign2VecPretraining, self).__init__(h5_fpath, transform)

    def __getitem__(self, idx):
        keypoints, _ = super(How2SignForSign2VecPretraining, self).__getitem__(idx)
        if self.skip_frames: keypoints = keypoints[::2]
        if self.max_sequence_length: keypoints = keypoints[: self.max_sequence_length]
        return {
            "input_values": keypoints,
        }
