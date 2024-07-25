import os
import sys
import h5py
import torch
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go two levels up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from sign2vec.modeling_sign2vec import Sign2VecModel
from sign2vec.feature_extraction_sign2vec import Sign2VecFeatureExtractor

import os
import h5py
import json
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sign2vec.modeling_sign2vec import Sign2VecModel
from tqdm import tqdm
from sign2vec.dataset.how2sign_hf5 import YoutubeASLForPretraining
from sign2vec.feature_extraction_sign2vec import Sign2VecFeatureExtractor

import argparse

def parse_args():
    model_name = "karahansahin/sign2vec-yasl-k10-c128-d0.5"
    data_dir = "pretraining/how2sign"
    output_path = "pretraining/how2sign"
    input_file = "H2S_train.h5"
    output_file = "sign2vec.train.0.h5"
    annotation_file = 'sign2vec/how2sign/h2s.annotations.train.json'
    metadata_file = 'metadata_sign2vec.train.json'

    parser = argparse.ArgumentParser()

    parser.add_argument('--layer_to_extract', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default=model_name)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--output_path', type=str, default=output_path)
    parser.add_argument('--input_file', type=str, default=input_file)
    parser.add_argument('--output_file', type=str, default=output_file)
    parser.add_argument('--annotation_file', type=str, default=annotation_file)
    parser.add_argument('--metadata_file', type=str, default=metadata_file)
    parser.add_argument('--use_shards', action='store_true')
    parser.add_argument('--shard_prefix', type=str, default='yasl_poses')
    
    return parser.parse_args()


def read_annotation_file(file_path, h5_file_path):

    with h5py.File(h5_file_path, 'r') as file:
        key2idx = {key: i for i, key in enumerate(file.keys())}

    with open(file_path, 'r') as file:
        annotation = json.load(file)

    video_ids = list(annotation.keys())
    dataset = []
    for video_id in video_ids:
        clip_ids = list(annotation[video_id]['clip_order'])
        for clip_id in clip_ids:
            dataset.append({
                'video_id': video_id,
                'clip_id': clip_id,
                'sentence_idx': key2idx[clip_id],
                'h5_file_path': h5_file_path,
            })

    return pd.DataFrame.from_records(dataset)

def transform_h5_to_pointer(annotation_csv):

    video_ids = []
    clip_ids = []
    sentence_ids = []
    h5_file_path = []
    current_video = None
    for row in annotation_csv.iterrows():
        row = row[1]
        if current_video != row['video_id']:
            current_video = row['video_id']
            video_ids.append(current_video)
            clip_ids.append([])
            sentence_ids.append([])
        
        clip_ids[-1].append(row['clip_id'])
        sentence_ids[-1].append(row['sentence_idx'])
        h5_file_path.append(row['h5_file_path'])

    return h5_file_path, video_ids, clip_ids, sentence_ids

def generate_metadata_file(metadata_file, video_ids, h5_file_idx):
    metadata = {video_id: h5_file_idx for video_id in video_ids}
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file)

def save_to_h5(fetures_list_h5, label, index_dataset, chunk_batch, chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        fetures_list_h5.resize(chunk_batch * chunk_size, axis=0)
    fetures_list_h5[index_dataset:index_dataset + chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

    args = parse_args()

    load_dotenv()
    token=os.environ.get("HUB_TOKEN")

    import os

    model = Sign2VecModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        token=token,
    )

    if args.use_shards:

        shards = [
            os.path.join(args.data_dir, file_name) for file_name in os.listdir(args.data_dir) if file_name.startswith(args.shard_prefix)
        ]

        annotation_file = os.path.join(args.output_path, args.annotation_file)
        with open(annotation_file, 'r') as file:
            annotation = json.load(file)
        
        req_ids = list(annotation.keys())

        print('tst',req_ids[:10])

        video_ids = []
        clip_ids = []
        sentence_ids = []
        fpaths = []
        annotation_csv = []
        current_video = None
        for shard in shards:
            with h5py.File(shard, 'r') as f:
                clips = list(f.keys())
                for idx, clip in enumerate(clips):
                    video_id = clip.split('.')[0]
                    print(video_id, req_ids[:10])
                    if video_id not in req_ids:
                        continue
                    if current_video != video_id:
                        current_video = video_id
                        video_ids.append(current_video)
                        clip_ids.append([])
                        sentence_ids.append([])
                    clip_ids[-1].append(clip)
                    sentence_ids[-1].append(idx)
                    fpaths.append(shard)
                    annotation_csv.append({
                        'video_id': video_id,
                        'clip_id': clip,
                        'sentence_idx': idx,
                        'h5_file_path': shard,
                    })
        
        print(annotation_csv[:10])
        
        dataset_type = args.output_file.split('.')[-2]
        metadata_file = f'metadata_sign2vec.{dataset_type}.0.json'
        with open(os.path.join(args.output_path, metadata_file), 'w') as file:
            json.dump({video_id: 0 for video_id in video_ids}, file)

        annotation_csv = pd.DataFrame.from_records(annotation_csv)
        annotation_csv.to_csv(os.path.join(args.output_path, 'annotation.csv'), index=False)
    else:
        annotation_csv = read_annotation_file(args.annotation_file, os.path.join(args.data_dir, args.input_file))

        annotation_csv.to_csv(os.path.join(args.output_path, 'annotation.csv'), index=False)

        fpaths, video_ids, clip_ids, sentence_ids = transform_h5_to_pointer(annotation_csv)

    generate_metadata_file(args.metadata_file, video_ids, args.output_file.split('.')[-2])

    train_data = YoutubeASLForPretraining(
        dataset=os.path.join(args.output_path, 'annotation.csv'),
        data_dir=args.data_dir,
        max_length=500,
        kp_norm=True,
        zero_mean_unit_var_norm=True,
        pose_version='full'
    )
    
    model.to(device)
        
    with h5py.File(os.path.join(args.output_path, args.output_file), 'a') as f_out:

        for i in tqdm(range(0, len(video_ids))):      # iterating over videos
            video = video_ids[i]
            video_h5 = f_out.create_group(video) if video not in f_out.keys() else f_out[video]
            for idx, clip in enumerate(clip_ids[i]):    # iterating over clips of the video
                # set starting index and starting chunk
                index_dataset = 0
                chunk_batch = 1

                features = train_data.get_pose_landmarks(fpaths[i], sentence_ids[i][idx])
                features = torch.tensor(features['input_values']).float()
                features = features.unsqueeze(0)
                input_features = features.transpose(1, 2)
                # number of samples in one chunk, set it to the same size as batch size during prediction
                chunk_size = 2

                # ADD PREDICTION OF YOUR MODEL HERE
                # pose landmarks are in list_of_features[i][idx]
                with torch.no_grad():
                    if input_features.shape[2] < 10:
                        input_features = torch.cat([input_features, torch.zeros((1, input_features.shape[1], 15 - input_features.shape[2] ))], dim=2)
                    try:
                        features = model(input_values=input_features.to(device), output_hidden_states=True)
                        if args.layer_to_extract > 0:
                            features = features.hidden_states[args.layer_to_extract].detach().cpu().numpy()[0]
                        else:
                            features = features.last_hidden_state.detach().cpu().numpy()[0]
                    except Exception as e:
                        print('Cannot extract features for video:', video, 'clip:', clip, 'error:', e, 'shape:', input_features.shape)
                        features = np.zeros((10, 768))

                features = features.astype(np.float16)
                fetures_list_h5 = video_h5.create_dataset(clip, shape=features.shape, maxshape=(None, features.shape[1]), dtype=np.dtype('float16'))
                num_full_chunks = len(features) // chunk_size
                last_chunk_size = len(features) % chunk_size
                for c in range(num_full_chunks):
                    feature = features[index_dataset:index_dataset + chunk_size]
                    index_dataset, chunk_batch = save_to_h5(fetures_list_h5, feature, index_dataset, chunk_batch,
                                                            chunk_size)
                if last_chunk_size > 0:
                    feature = features[index_dataset:index_dataset + last_chunk_size]
                    index_dataset, chunk_batch = save_to_h5(fetures_list_h5, feature, index_dataset, chunk_batch,
                                                            last_chunk_size)