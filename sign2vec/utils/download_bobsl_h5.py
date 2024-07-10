import os
import sys
import tarfile
import numpy as np
from tqdm import tqdm
import h5py

from dotenv import load_dotenv
load_dotenv()

def extract_tar(tar_file, dest_dir='tmp'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    with tarfile.open(tar_file, 'r') as tar:
        tar.extractall(dest_dir)
    
    return dest_dir, tar_file

def write_to_hdf5(file_id, h5file_filename, data):
    with h5py.File(h5file_filename, 'a') as f:
        try: f.create_group(file_id)
        except: pass

        try: f.create_group(f'{file_id}/joints')
        except: pass

        for k in data.keys():
            f.create_dataset(f'{file_id}/joints/{k}', data=data[k])

    return True

def download_bobsl_video(username, password, video_url, video_path):

    if not video_url.startswith("http"):
        return None
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    username = os.getenv("BOBSL_USERNAME", username)
    password = os.getenv("BOBSL_PASSWORD", password)

    if username is None or password is None:
        raise Exception(
            "Please set BOBSL_USERNAME and BOBSL_PASSWORD environment variables"
        )

    os.system(
        f"curl --user {username}:{password} {video_url} --output {video_path}"
    )

def extract_video_frames(output_dir):

    files = os.listdir(output_dir)
    file_id = files[0].split('_')[0]
    # get json files only
    files = [os.path.join(output_dir, f) for f in files if f.endswith(".json")]

    if len(files) == 0:
        raise Exception("No frames extracted")
    
    sorted_files = sorted(files)

    keypoints = {
        'face_landmarks': [],
        'pose_keypoints': [],
        'hand_left_keypoints': [],
        'hand_right_keypoints': [],
        'indices': []
    }
    for f in tqdm(sorted_files):
        import json
        with open(f, 'r') as file:
            frame_id = int(f.split('.')[0].split('_')[-2])
            try:
                data = json.load(file)
                if len(data['people']) == 0:
                    continue
                person = data['people'][0]

                if 'pose_keypoints_2d' not in person:
                    continue

                keypoints['face_landmarks'].append(np.array(person['face_keypoints_2d']).reshape(-1, 3)) 
                keypoints['pose_keypoints'].append(np.array(person['pose_keypoints_2d']).reshape(-1, 3)) 
                keypoints['hand_left_keypoints'].append(np.array(person['hand_left_keypoints_2d']).reshape(-1, 3)  ) 
                keypoints['hand_right_keypoints'].append(np.array(person['hand_right_keypoints_2d']).reshape(-1, 3)) 
                keypoints['indices'].append(frame_id)

            except Exception as e:
                print(e,'for frame', frame_id)
                continue

    for k in keypoints:
        keypoints[k] = np.array(keypoints[k])

    return file_id, keypoints

def download_bobsl(
    url_file='bobsl.txt',
    batch_size=100,
    BOBSL_USERNAME=None,
    BOBSL_PASSWORD=None,
    tmp_path='tmp',
    save_dir='data'
):
    
    with open(url_file, 'r') as f:
        
        lines = f.readlines()
        batch_idx = 0
        for idx, line in enumerate(lines):

            if idx+1 % batch_size == 0:
                batch_idx += 1
            
            h5file_filename = f'BOBSL_train_{batch_idx}.h5'

            h5file_filename = os.path.join(save_dir, h5file_filename)

            video_url = line.strip()
            if not video_url:
                continue
            video_path = video_url.split('/')[-1]
            video_path = os.path.join(tmp_path, video_path)

            
            download_bobsl_video(BOBSL_USERNAME, BOBSL_PASSWORD, video_url, video_path)
            print(f"Downloaded {video_path}")
            extract_tar(video_path, tmp_path)
            print(f"Extracted {video_path}")
            file_id, keypoints = extract_video_frames(tmp_path)
            print(f"Extracted {len(keypoints['indices'])} frames")
            write_to_hdf5(file_id, h5file_filename, keypoints)

            # remove video
            os.remove(video_path)
            # remove tmp folder
            os.system('rm -rf tmp')

            print(f"Saved {h5file_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--url_file', type=str, default='bobsl.txt')
    parser.add_argument('--batch_size_per_h5', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='data')
    args = parser.parse_args()

    download_bobsl(
        url_file=args.url_file,
        batch_size=args.batch_size_per_h5,
        save_dir=args.save_dir
    )