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
                 feature_extractor=None,
                 add_adapter=True
                 ):
        
        self.use_pose = use_pose
        self.use_hands = use_hands
        self.use_face = use_face
        self.max_frame_diff = max_frame_diff
        self.add_adapter = add_adapter

        self.max_length = max_length
        self.fps = fps

        self.max_frames = int(max_length / (sampling_rate / fps))

        print('BOBSLDataset')
        print('Sampling rate:', sampling_rate)
        print('Max frame diff:', max_frame_diff)
        print('Max length:', max_length)
        print('FPS:', fps)
        print('Max Frames:', self.max_frames)
        print('Info path:', info_path)
        print('Data path:', data_path)

        self.stride = stride
        self.sampling_rate = sampling_rate 

        self.data_path = data_path
        self.info_path = info_path
        self.feature_extractor = feature_extractor
        self.data = self.load_data()

        print(f'Loaded {len(self.data)} training samples')

    def load_data(self):
        data = []
        with open(self.info_path, 'r') as file:
            metadata = json.load(file)

        for DOC in metadata:

            doc_id = DOC['document_id']
            frame_ids = DOC['frame_ids']
            frame_ids = np.array(frame_ids).astype(int)
            frame_diff = np.diff(frame_ids)

            array = np.load(f'{self.data_path}/{doc_id}.npy').reshape(-1)

            for i in list(range(0, len(frame_ids) - self.max_frames, self.stride)):
                
                frames = frame_ids[i:i+self.max_frames]

                joint_per_frame = (self.sampling_rate / self.fps)

                start_time = int( i * joint_per_frame )
                end_time = int( (i + self.max_frames) * joint_per_frame )

                if not array[start_time:end_time].shape[0]:
                    print('Array is empty')
                    continue

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

        array = array[start_time:end_time]

        array = array.reshape(
            int(self.sampling_rate / self.fps),
            -1
        )

        return {
            'input_values': array,
        }
