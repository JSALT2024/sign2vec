import os
import cv2
import h5py
import torch
import datetime
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, Dataset
from sign2vec.utils.normalization import local_keypoint_normalization, global_keypoint_normalization

class How2SignDatasetForFinetuning(Dataset):


    def __init__(self, 
                 dataset, 
                 max_length=25*20,
                 feature_size=167,
                 sampling_rate=25,
                 data_dir=None,
                 padding_value=0.0):
        
        self.data_dir = data_dir
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
        )

        self.max_length = max_length
        self.dataset = pd.read_csv(dataset)
        self.dataset.dropna(inplace=True)
        self.loader = How2SignDataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        h5_path = os.path.join( self.data_dir ,self.dataset['h5_file_path'].iloc[idx])
        sentence_idx = self.dataset['sentence_idx'].iloc[idx]
        dataset = self.loader(h5_path)
        
        data, sentence = dataset.load_data(idx=sentence_idx)
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data
        
        # Select only the landmarks that are relevant for the sign language
        pose_landmarks = pose_landmarks[:, [11, 12, 13, 14, 23, 24], :]
        # Add min-max scaling to each landmark
        data = np.concatenate([pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks], axis=1)

        x_vals = data[:, 0].reshape(-1, 1)
        y_vals = data[:, 1].reshape(-1, 1)

        try:
            data_min_x_nonzero = np.min(x_vals[x_vals != 0])
            data_max_x = np.max(x_vals)
        except:
            data_min_x_nonzero = 0
            data_max_x = np.max(x_vals)

        try:
            data_min_y_nonzero = np.min(y_vals[y_vals != 0])
            data_max_y = np.max(y_vals)
        except:
            data_min_y_nonzero = 0
            data_max_y = np.max(y_vals)

        # data = (data - [data_min_x_nonzero, data_min_y_nonzero, data_min_z_nonzero]) / [data_max_x, data_max_y, data_max_z]
        data[:,:,:2] = (data[:,:,:2] - [data_min_x_nonzero, data_min_y_nonzero]) / [data_max_x, data_max_y]
        data = torch.tensor(data).reshape(data.shape[0], -1)

        data = torch.tensor(data).reshape(data.shape[0], -1)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'input_values': data,
            'sentence': sentence,
        }


class How2SignDatasetForPretraining(Dataset):

    def __init__(self, 
                 dataset, 
                 max_length=25*20, 
                 data_dir=None,
                 padding="max_length"):
        
        self.data_dir = data_dir
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=167,
            sampling_rate=25,
            padding_value=0.0,
        )
        
        self.max_length = max_length
        self.dataset = pd.read_csv(dataset)
        self.dataset.dropna(inplace=True)
        self.dataset[self.dataset['video_path'].apply(lambda x: True if x else False)]
        self.loader = How2SignDataset

    def __len__(self):
        return len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        h5_path = os.path.join( self.data_dir ,self.dataset['h5_file_path'].iloc[idx])
        sentence_idx = self.dataset['sentence_idx'].iloc[idx]
        dataset = self.loader(h5_path)

        data, sentence = dataset.load_data(idx=sentence_idx)
        
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data

        # Add min-max scaling to each landmark
        pose_landmarks

        face_landmarks = face_landmarks.reshape(-1, 32*2)
        pose_landmarks = pose_landmarks.reshape(-1, 33,*2)
        right_hand_landmarks = right_hand_landmarks.reshape(-1, 21*2)
        left_hand_landmarks = left_hand_landmarks.reshape(-1, 21*2)
        
        data = np.concatenate([
            pose_landmarks, 
            right_hand_landmarks, 
            left_hand_landmarks, 
            face_landmarks
        ], axis=1)

        data = torch.tensor(data).reshape(data.shape[0], -1)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        data = data / data.max()

        return {
            'input_values': data
        }


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

        self.face_landmarks = [
            0, 4, 13, 14, 17, 33, 37, 
            39, 46, 52, 55, 61, 64, 81,
            82, 93, 133, 151, 152, 
            159, 172, 178, 181, 263, 269, 276, 
            282, 285, 291, 294, 311, 323, 362, 
            386, 397, 468, 473
            ]
        self.transform = transform
        self.kp_normalization = kp_normalization

        self.get_video_names()

    def __getitem__(self, index):

        data, sentence = self.load_data(idx=index)
        if self.transform:
            data = self.transform(data)

        return {"data": torch.tensor(data).float(), "sentence": sentence}

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            return len(f.keys())

    def get_video_names(self):
        """
        Get video names for .mp4 videos in self.video_path dir.
        returns:
            video_paths (list): list of .mp4 files available in self.video_path
        """
        if self.video_path is None:
            return

        if not os.path.exists(self.video_path):
            raise ValueError(f'Error: video_path does not exist \n {self.video_path}')
        if not os.path.isdir(self.video_path):
            raise ValueError(f'Error: video_path is not a directory \n {self.video_path} ')

        for filename in os.listdir(self.video_path):
            if filename.endswith(".mp4"):
                self.video_names[filename.strip('.mp4')] = os.path.join(self.video_path, filename)

    def load_data(self, idx=0):

        with h5py.File(self.h5_path, 'r') as f:
            video_name = list(f.keys())[idx]
            joints = {l[0]: l[1][()] for l in f[video_name]['joints'].items()}
            face_landmarks = f[video_name]['joints']['face_landmarks'][()]
            left_hand_landmarks = f[video_name]['joints']['left_hand_landmarks'][()]
            right_hand_landmarks = f[video_name]['joints']['right_hand_landmarks'][()]
            pose_landmarks = f[video_name]['joints']['pose_landmarks'][()]
            sentence = f[video_name]['sentence'][()].decode('utf-8')

        # TODO implement once visual training is relevant
        # if self.video_path is None:
        #     pass
        # elif video_name in self.video_names:
        #     video_path = os.path.join(self.video_path + video_name + '.mp4')
        # else:
        #     warnings.warn(
        #         f'Warning: video_path does not contain video with name {video_name}')

        if self.kp_normalization:
            joints["face_landmarks"] = joints["face_landmarks"][:, self.face_landmarks, :]

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
            if additional_landmarks:
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

            data = np.concatenate(data, axis=1)
            data = data.reshape(data.shape[0], -1)
        else:
            face_landmarks = face_landmarks[:, self.face_landmarks, :3]  # select only wanted KPI and  x, y
            left_hand_landmarks = left_hand_landmarks[:, :, :3]
            right_hand_landmarks = right_hand_landmarks[:, :, :3]
            pose_landmarks = pose_landmarks[:, :, :3]

            data = np.concatenate((pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks),
                                  axis=1)
            # data = np.concatenate((pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks),
            #                       axis=1).reshape(len(face_landmarks), 214)

        return (pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks), sentence

    def plot_points2video(self, index, video_name):
        if self.video_path is None:
            raise ValueError("Error: video_path is None, cannot plot. \n Aborting.")
        item = self.__getitem__(index)
        plot_metadata = item['plot_metadata']

        cap = cv2.VideoCapture(item['metadata']['VIDEO_PATH'])

        # Check if the video file opened successfully
        if not cap.isOpened():
            raise ValueError(f'Error: Couldnt open the video file. \n {video_path} \n Aborting.')

        ret, frame = cap.read()

        height, width, layers = frame.shape
        idx = 0
        video = cv2.VideoWriter(video_name, 0, 3, (width, height))

        while ret:
            frame = self.anotate_img(frame, plot_metadata, idx, (125, 255, 10))
            video.write(frame)
            ret, frame = cap.read()
            idx += 1

        cap.release()
        cv2.destroyAllWindows()
        video.release()