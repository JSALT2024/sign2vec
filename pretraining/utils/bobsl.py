import json 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

class BOBSLDataset(Dataset):

    def __init__(self, 
                 use_pose=True,
                 use_hands=True,
                 use_face=False,
                 sampling_rate=411*25,
                 max_frame_diff=10,
                 max_length=1000,
                 fps=25,
                 stride=1,
                 info_path='info.json',
                 data_path='data/bobsl', 
                 feature_extractor=None
                 ):
        
        self.use_pose = use_pose
        self.use_hands = use_hands
        self.use_face = use_face
        self.max_frame_diff = max_frame_diff

        self.max_length = max_length

        self.max_frames = int(max_length / (sampling_rate / fps))
        self.stride = stride
        self.sampling_rate = sampling_rate

        self.data_path = data_path
        self.info_path = info_path
        self.feature_extractor = feature_extractor
        self.data = self.load_data()

        print(f'Loaded {len(self.data)} samples')
        print(f'With max_frames: {self.max_frames} and stride: {self.stride}')

    def load_data(self):
        data = []
        with open(self.info_path, 'r') as file:
            metadata = json.load(file)

        for DOC in metadata:

            doc_id = DOC['document_ids']
            frame_ids = DOC['frame_ids']
            frame_ids = np.array(frame_ids).astype(int)
            frame_diff = np.diff(frame_ids)

            for i in range(0, len(frame_ids) - self.max_frames, self.stride):

                if frame_diff[i:i+self.max_frames].max() < self.max_frame_diff:

                    frames = frame_ids[i:i+self.max_frames]
                    start_time = int(i * self.sampling_rate)
                    end_time = int((i+self.max_frames) * self.sampling_rate)
                    data.append({
                        'doc_id': doc_id,
                        'frame_ids': frames,
                        'start_time': start_time,
                        'end_time': end_time
                    })

        return data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        doc_id = self.data[idx]['doc_id']
        start_time = self.data[idx]['start_time']
        end_time = self.data[idx]['end_time']
        
        array = np.load(f'{self.data_path}/{doc_id}.npy')

        inputs = self.feature_extractor(
            array[start_time:end_time], max_length=self.max_length, truncation=True, sampling_rate=self.sampling_rate
        )

        return {
            'input_values': inputs.input_values[0],
            # 'input_length': len(inputs.input_values[0]),
            # 'start_time': start_time,
            # 'end_time': end_time,
        }
