import os
import h5py
import torch
import datetime
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, Dataset
from sign2vec.utils.normalization import local_keypoint_normalization, global_keypoint_normalization

class How2SignForFinetuning(Dataset):

    def __init__(self, 
                 dataset, 
                 max_length=25*20, 
                 data_dir=None,
                 kp_norm=None,
                 zero_mean_unit_var_norm=None,
                 add_noise=False,
                 pose_version="full",
                 padding="max_length"):
        
        self.data_dir = data_dir

        self.face_landmarks = [
            0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81, 
            93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
            282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
        ]
        if pose_version == "full":
            self.pose_landmarks = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 ]
        if pose_version == "yasl":
            self.pose_landmarks = [ 11, 12, 13, 14, 23, 24 ]

        # Maybe not normalize face landmarks
        # Extreme change in face landmarks can cause the model gradient to explode
        # Look at codebooks (pose, right_hand, left_hand, face)
        # if orientation to the camera (if there is think about rotation) (chest triangle - pose rotation)
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=340,
            sampling_rate=25,
            padding_value=0.0,
            do_normalize=zero_mean_unit_var_norm
        )

        self.add_noise = add_noise
        self.kp_norm = kp_norm
        self.norm = [
            "global-pose_landmarks",
            "local-right_hand_landmarks", # "local-right_hand_landmarks",
            "local-left_hand_landmarks", # "local-left_hand_landmarks",
            "local-face_landmarks"
        ]
            
        self.max_length = max_length
        self.dataset = pd.read_csv(dataset)
        self.dataset.dropna(inplace=True)
        self.loader = How2SignDataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        h5_path = os.path.join( self.data_dir ,self.dataset['h5_file_path'].iloc[idx])
        sentence_idx = self.dataset['sentence_idx'].iloc[idx]
        
        dataset = self.loader(h5_path, kp_normalization=self.norm if self.kp_norm else [])
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks, sentence = dataset.load_data(idx=sentence_idx)

        sentence = sentence.decode("utf-8")

        pose_landmarks = pose_landmarks[:, self.pose_landmarks, :]
        face_landmarks = face_landmarks[:, self.face_landmarks, :]

        face_landmarks = face_landmarks.reshape( face_landmarks.shape[0], -1 )
        pose_landmarks = pose_landmarks.reshape( pose_landmarks.shape[0], -1 )
        right_hand_landmarks = right_hand_landmarks.reshape( right_hand_landmarks.shape[0], -1 )
        left_hand_landmarks = left_hand_landmarks.reshape( left_hand_landmarks.shape[0], -1 )

        data = np.concatenate([pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks], axis=1)
        data = torch.tensor(data).reshape(data.shape[0], -1)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        data = self.feature_extractor(
            data, 
            max_length=self.max_length, 
            truncation=True, 
            sampling_rate=25,
        )
        
        data = data['input_values'][0]
        
        return {
            'input_values': data,
            "sentence": sentence
        }

class YoutubeASLForPretraining(Dataset):

    def __init__(self, 
                 dataset, 
                 max_length=25*20, 
                 data_dir=None,
                 kp_norm=None,
                 zero_mean_unit_var_norm=None,
                 add_noise=False,
                 pose_version="full",
                 padding="max_length"):
        
        self.data_dir = data_dir

        self.face_landmarks = [
            0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81, 
            93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
            282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
        ]
        if pose_version == "full":
            self.pose_landmarks = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 ]
        if pose_version == "yasl":
            self.pose_landmarks = [ 11, 12, 13, 14, 23, 24 ]

        # Maybe not normalize face landmarks
        # Extreme change in face landmarks can cause the model gradient to explode
        # Look at codebooks (pose, right_hand, left_hand, face)
        # if orientation to the camera (if there is think about rotation) (chest triangle - pose rotation)
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=340,
            sampling_rate=25,
            padding_value=0.0,
            do_normalize=zero_mean_unit_var_norm
        )

        self.add_noise = add_noise
        self.kp_norm = kp_norm
        self.norm = [
            "global-pose_landmarks",
            "local-right_hand_landmarks", # "local-right_hand_landmarks",
            "local-left_hand_landmarks", # "local-left_hand_landmarks",
            "local-face_landmarks"
        ]
            
        self.max_length = max_length
        self.dataset = pd.read_csv(dataset)
        self.dataset.dropna(inplace=True)
        self.loader = YoutubeASL

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        h5_path = os.path.join( self.data_dir ,self.dataset['h5_file_path'].iloc[idx])
        sentence_idx = self.dataset['sentence_idx'].iloc[idx]
        
        dataset = self.loader(h5_path, kp_normalization=self.norm if self.kp_norm else [])
        data = dataset.load_data(idx=sentence_idx)
        
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data

        pose_landmarks = pose_landmarks[:, self.pose_landmarks, :]
        face_landmarks = face_landmarks[:, self.face_landmarks, :]

        face_landmarks = face_landmarks.reshape( face_landmarks.shape[0], -1 )
        pose_landmarks = pose_landmarks.reshape( pose_landmarks.shape[0], -1 )
        right_hand_landmarks = right_hand_landmarks.reshape( right_hand_landmarks.shape[0], -1 )
        left_hand_landmarks = left_hand_landmarks.reshape( left_hand_landmarks.shape[0], -1 )

        data = np.concatenate([pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks], axis=1)
        data = torch.tensor(data).reshape(data.shape[0], -1)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        data = self.feature_extractor(
            data, 
            max_length=self.max_length, 
            truncation=True, 
            sampling_rate=25,
        )
        
        data = data['input_values'][0]
        
        return {
            'input_values': data,
        }
    
    def get_pose_landmarks(self, h5_path, sentence_idx):
        
        dataset = self.loader(h5_path, kp_normalization=self.norm if self.kp_norm else [])
        data, sentence = dataset.load_data(idx=sentence_idx)
        
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data

        pose_landmarks = pose_landmarks[:, self.pose_landmarks, :]
        face_landmarks = face_landmarks[:, self.face_landmarks, :]

        face_landmarks = face_landmarks.reshape( face_landmarks.shape[0], -1 )
        pose_landmarks = pose_landmarks.reshape( pose_landmarks.shape[0], -1 )
        right_hand_landmarks = right_hand_landmarks.reshape( right_hand_landmarks.shape[0], -1 )
        left_hand_landmarks = left_hand_landmarks.reshape( left_hand_landmarks.shape[0], -1 )

        data = np.concatenate([pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks], axis=1)
        data = torch.tensor(data).reshape(data.shape[0], -1)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        data = self.feature_extractor(
            data, 
            max_length=self.max_length, 
            truncation=True, 
            sampling_rate=25,
        )
        
        data = data['input_values'][0]

        return {
            'input_values': data,
            'sentence': sentence
        }
    
    def get_raw_landmarks(self, h5_path, sentence_idx):
        
        dataset = self.loader(h5_path, kp_normalization=self.norm if self.kp_norm else [])
        data = dataset.load_data(idx=sentence_idx)
        
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data

        pose_landmarks = pose_landmarks[:, self.pose_landmarks, :]
        face_landmarks = face_landmarks[:, self.face_landmarks, :]

        face_landmarks = face_landmarks.reshape( face_landmarks.shape[0], -1 )
        pose_landmarks = pose_landmarks.reshape( pose_landmarks.shape[0], -1 )
        right_hand_landmarks = right_hand_landmarks.reshape( right_hand_landmarks.shape[0], -1 )
        left_hand_landmarks = left_hand_landmarks.reshape( left_hand_landmarks.shape[0], -1 )

        data = np.concatenate([pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks], axis=1)
        data = torch.tensor(data).reshape(data.shape[0], -1)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data.detach().numpy()

class YoutubeASL(Dataset):
    """ Custom dataset for how2sign dataset on pose features.
    args:
        h5_path (str): path to h5 file
        video_file_path (str, optional): path to video files
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, h5_path, video_file_path=None, transform=None, kp_normalization: list = []):

        self.video_path = video_file_path
        self.h5_path = h5_path

        self.video_names = {}
        self.transform = transform
        self.kp_normalization = kp_normalization

    def __getitem__(self, index):

        data = self.load_data(idx=index)
        if self.transform:
            data = self.transform(data)

        return {"data": torch.tensor(data).float()}

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            return len(f.keys())

    def load_data(self, idx=0):

        with h5py.File(self.h5_path, 'r') as f:
            video_name = list(f.keys())[idx]
            joints = {l[0]: l[1][()] for l in f[video_name]['joints'].items()}
            face_landmarks = f[video_name]['joints']['face_landmarks'][()]
            left_hand_landmarks = f[video_name]['joints']['left_hand_landmarks'][()]
            right_hand_landmarks = f[video_name]['joints']['right_hand_landmarks'][()]
            pose_landmarks = f[video_name]['joints']['pose_landmarks'][()]

        if self.kp_normalization:
            local_landmarks = {}
            global_landmarks = {}

            for idx, landmarks in enumerate(self.kp_normalization):
                prefix, landmarks = landmarks.split("-")
                if prefix == "local":
                    local_landmarks[idx] = landmarks
                elif prefix == "global":
                    global_landmarks[idx] = landmarks

            # local normalization
            for idx, landmarks in local_landmarks.items():
                normalized_keypoints = local_keypoint_normalization(joints, landmarks, padding=0.2)
                local_landmarks[idx] = normalized_keypoints

            # global normalization
            additional_landmarks = list(global_landmarks.values())
            if "pose_landmarks" in additional_landmarks:
                additional_landmarks.remove("pose_landmarks")
            keypoints, additional_keypoints = global_keypoint_normalization(
                joints,
                "pose_landmarks",
                additional_landmarks
            )
            for k,  landmark in global_landmarks.items():
                if landmark == "pose_landmarks":
                    global_landmarks[k] = keypoints
                else:
                    global_landmarks[k] = additional_keypoints[landmark]

            all_landmarks = {**local_landmarks, **global_landmarks}
            data = []
            for idx in range(len(self.kp_normalization)):
                data.append(all_landmarks[idx])

            return data[0], data[1], data[2], data[3]

        face_landmarks = face_landmarks[:, :, :2]  # select only wanted KPI and  x, y
        left_hand_landmarks = left_hand_landmarks[:, :, :2]
        right_hand_landmarks = right_hand_landmarks[:, :, :2]
        pose_landmarks = pose_landmarks[:, :, :2]

        return (pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks) 


class How2SignDataset(Dataset):
    """ Custom dataset for how2sign dataset on pose features.
    args:
        h5_path (str): path to h5 file
        video_file_path (str, optional): path to video files
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, h5_path, video_file_path=None, transform=None, kp_normalization: list = []):

        self.video_path = video_file_path
        self.h5_path = h5_path

        self.video_names = {}
        self.transform = transform
        self.kp_normalization = kp_normalization

    def __getitem__(self, index):

        data = self.load_data(idx=index)
        if self.transform:
            data = self.transform(data)

        return {"data": torch.tensor(data).float()}

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            return len(f.keys())

    def load_data(self, idx=0):

        with h5py.File(self.h5_path, 'r') as f:
            video_name = list(f.keys())[idx]
            joints = {l[0]: l[1][()] for l in f[video_name]['joints'].items()}
            face_landmarks = f[video_name]['joints']['face_landmarks'][()]
            left_hand_landmarks = f[video_name]['joints']['left_hand_landmarks'][()]
            right_hand_landmarks = f[video_name]['joints']['right_hand_landmarks'][()]
            pose_landmarks = f[video_name]['joints']['pose_landmarks'][()]
            sentence = f[video_name]['sentence'][()]

        if self.kp_normalization:
            local_landmarks = {}
            global_landmarks = {}

            for idx, landmarks in enumerate(self.kp_normalization):
                prefix, landmarks = landmarks.split("-")
                if prefix == "local":
                    local_landmarks[idx] = landmarks
                elif prefix == "global":
                    global_landmarks[idx] = landmarks

            # local normalization
            for idx, landmarks in local_landmarks.items():
                normalized_keypoints = local_keypoint_normalization(joints, landmarks, padding=0.2)
                local_landmarks[idx] = normalized_keypoints

            # global normalization
            additional_landmarks = list(global_landmarks.values())
            if "pose_landmarks" in additional_landmarks:
                additional_landmarks.remove("pose_landmarks")
            keypoints, additional_keypoints = global_keypoint_normalization(
                joints,
                "pose_landmarks",
                additional_landmarks
            )
            for k,  landmark in global_landmarks.items():
                if landmark == "pose_landmarks":
                    global_landmarks[k] = keypoints
                else:
                    global_landmarks[k] = additional_keypoints[landmark]

            all_landmarks = {**local_landmarks, **global_landmarks}
            data = []
            for idx in range(len(self.kp_normalization)):
                data.append(all_landmarks[idx])

            pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data

        face_landmarks = face_landmarks[:, :, :2]  # select only wanted KPI and  x, y
        left_hand_landmarks = left_hand_landmarks[:, :, :2]
        right_hand_landmarks = right_hand_landmarks[:, :, :2]
        pose_landmarks = pose_landmarks[:, :, :2]

        return pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks, sentence