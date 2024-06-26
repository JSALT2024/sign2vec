import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import json
import glob

class How2SignDatasetForPretraining(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 keypoint_path: str,
                 tokenizer: AutoTokenizer,
                 max_frames: int = 250,
                 ) -> None:
        super().__init__()

        self.dataframe = dataframe
        self.keypoint_path = keypoint_path
        self.tokenizer = tokenizer
        self.max_frames = max_frames

    def __len__(self):
        return self.dataframe.shape[0]
    

    def __getitem__(self, idx: int) -> dict:
        row = self.dataframe.iloc[idx]

        keypoints = {
            'pose': [],
            'face': [],
            'left_hand': [],
            'right_hand': []
        }
        sorted_files = sorted(glob.glob(row['POSE_PATH']+'*.json'))
        for file in sorted_files:
            with open(file) as f:
                for person in json.load(f)['people']:
                    keypoints['pose'].append(person['pose_keypoints_2d'])
                    keypoints['face'].append(person['face_keypoints_2d'])
                    keypoints['left_hand'].append(person['hand_left_keypoints_2d'])
                    keypoints['right_hand'].append(person['hand_right_keypoints_2d'])
        
        # truncate or pad to max_frames
        keypoints['face'] = keypoints['face'][:self.max_frames]
        keypoints['pose'] = keypoints['pose'][:self.max_frames]
        keypoints['left_hand'] = keypoints['left_hand'][:self.max_frames]
        keypoints['right_hand'] = keypoints['right_hand'][:self.max_frames]

        # convert to tensor
        pose = torch.tensor(keypoints['pose']) # (T, 75)
        face = torch.tensor(keypoints['face'])
        left_hand = torch.tensor(keypoints['left_hand'])
        right_hand = torch.tensor(keypoints['right_hand'])

        # # reshape to (T, N, C)
        # pose = pose.view(-1, 25, 3)
        # face = face.view(-1, 70, 3)
        # left_hand = left_hand.view(-1, 21, 3)
        # right_hand = right_hand.view(-1, 21, 3)

        # concatenate all keypoints
        keypoints = torch.cat([
            pose, face, left_hand, right_hand
        ], dim=1)

        return {
            'keypoints': keypoints,
            'text': row['SENTENCE'],
            'keypoints_path': row['SENTENCE_NAME'],
            'frames': len(sorted_files),
            'decoder_input_ids': self.tokenizer(row['SENTENCE'], return_tensors='pt').input_ids
        }


import os
import pandas as pd
import matplotlib.pyplot as plt

def check_file_exists(x): return len(glob.glob(x+'*.json'))

def get_how2sign_dataset(DATASET_PATH = 'how2sign/', verbose=False):
    """
    Load the How2Sign dataset and return the train, val, test
    splits as dataframes.
    
    Args:
        DATASET_PATH: str
        verbose: bool

    Returns:
        train_df: pd.DataFrame
        val_df: pd.DataFrame
        test_df: pd.DataFrame    
    """

    train_df = pd.read_csv(
        os.path.join(DATASET_PATH, 'labels','how2sign_realigned_train.csv'), sep='\t'
    )

    val_df = pd.read_csv(
        os.path.join(DATASET_PATH, 'labels','how2sign_val.csv'), sep='\t'
    )

    test_df = pd.read_csv(
        os.path.join(DATASET_PATH, 'labels','how2sign_realigned_test.csv'), sep='\t'
    )

    train_df['POSE_PATH'] = train_df['SENTENCE_NAME'].apply(lambda x: os.path.join(DATASET_PATH,f'pose/train/json/{x}/'))
    val_df['POSE_PATH'] = val_df['SENTENCE_NAME'].apply(lambda x: os.path.join(DATASET_PATH,f'pose/val/json/{x}/'))
    test_df['POSE_PATH'] = test_df['SENTENCE_NAME'].apply(lambda x: os.path.join(DATASET_PATH,f'pose/test/json/{x}/'))
    
    train_df['FRAMES'] = train_df['POSE_PATH'].apply(check_file_exists)
    val_df['FRAMES'] = val_df['POSE_PATH'].apply(check_file_exists)
    test_df['FRAMES'] = test_df['POSE_PATH'].apply(check_file_exists)

    if verbose:

        test_df.FRAMES.plot(kind='hist', bins=100)

        plt.figure(figsize=(20, 5))
        val_df.FRAMES.plot(kind='hist', bins=100)

        plt.show()

    return train_df, val_df, test_df