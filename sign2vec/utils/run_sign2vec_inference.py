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


def save_to_h5(fetures_list_h5, label, index_dataset, chunk_batch, chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        fetures_list_h5.resize(chunk_batch * chunk_size, axis=0)
    fetures_list_h5[index_dataset:index_dataset + chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch

def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='Sign2Vec Inference')

    parser.add_argument('--input_path', type=str, default='h5py', help='Path to the input h5 file' )
    parser.add_argument('--input_file', type=str, default='my_h5_file.h5', help='Name of the input h5 file' )
    parser.add_argument('--output_path', type=str, default='h5py', help='Path to the output h5 file' )
    parser.add_argument('--output_file', type=str, default='my_h5_file.h5', help='Name of the output h5 file' )
    parser.add_argument('--model_name', type=str, default='karahansahin/sign2vec-yasl-c64-m50-d0.5', help='Name of the model' )
    parser.add_argument('--dataset_name', type=str, default='yasl', help='Name of the dataset' )

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()

    # h5py file initialization
    f_in = h5py.File(os.path.join(args.input_path, args.input_file), 'r')
    video_ids = list(f_in.keys())

    load_dotenv()
    token=os.environ.get("HUB_TOKEN")

    model = Sign2VecModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        token=token,
    )
    
    feature_extractor = Sign2VecFeatureExtractor(
        feature_size=170,
        sampling_rate=25,
        max_duration_in_seconds=20.0,
        kp_norm=True,
        do_normalize=True,
    )

    # h5py file initialization4,
    f_out = h5py.File(os.path.join(args.output_path, args.output_file), 'w')

    # special data type for numpy array with variable length
    dt = h5py.vlen_dtype(np.dtype('float64'))

    for i in tqdm(range(0, len(video_ids))):      # iterating over videos
        
        if args.dataset_name == 'how2sign':
            video = video_ids[i]
            video_h5 = f_out.create_group(video)

            # set starting index and starting chunk
            index_dataset = 0
            chunk_batch = 1

            # number of samples in one chunk, set it to the same size as batch size during prediction
            chunk_size = 2

            # ADD PREDICTION OF YOUR MODEL HERE
            # pose landmarks are in list_of_features[i][idx]
            pose_landmarks = {l[0]: l[1][()] for l in f_in[video]['joints'].items()}
            features = feature_extractor(
                pose_landmarks=pose_landmarks
            )
            arr = features['input_values'][0]
            arr = np.moveaxis(arr, 1, 2)
            print('Input array shape:',arr.shape) # (2, 170, 25)
            with torch.no_grad():
                features = model(input_values=torch.tensor(arr).float()).last_hidden_state.detach().numpy()
                features = features[0]

            print('Output vec:',features[0].shape) # (2, 170, 768)

            fetures_list_h5 = video_h5.create_dataset('features', shape=(len(features),), maxshape=(None,), dtype=dt)
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


        if args.dataset_name == 'yasl':
            video, clip = video_ids[i].split('.')
            video_h5 = f_out.create_group(video)

            # set starting index and starting chunk
            index_dataset = 0
            chunk_batch = 1

            # number of samples in one chunk, set it to the same size as batch size during prediction
            chunk_size = 2

            # ADD PREDICTION OF YOUR MODEL HERE
            # pose landmarks are in list_of_features[i][idx]
            pose_landmarks = f_in[video][clip][()]
            features = feature_extractor(
                pose_landmarks=pose_landmarks
            )
            arr = features['input_values'][0]
            arr = np.moveaxis(arr, 1, 2)
            with torch.no_grad():
                features = model(input_values=torch.tensor(arr).float()).last_hidden_state.detach().numpy()
        
            fetures_list_h5 = video_h5.create_dataset(clip, shape=(len(features),), maxshape=(None,), dtype=dt)
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

    f_out.close()