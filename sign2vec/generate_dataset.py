import os
import glob
import json
import pickle
import json
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)

POSE_COLS = [
    'pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d'
]

def merge_keypoints(row):
    return np.concatenate([row[pose_col] for pose_col in POSE_COLS])


def download_video(video_url, username, password):

    if not video_url.startswith("http"):
        return None
    
    video_url = video_url.replace("\n", "")
    video_name = video_url.split("/")[-1].strip().replace("\n", "")
    video_path = f"tmp/{video_name}"
    os.makedirs("tmp/", exist_ok=True)
    document_id = video_name.split("_")[0].strip().split(".")[0]
    
    print(f"Processing {document_id}")

    if os.path.exists(f'features/{document_id}.pkl'):
        print(f'Ignoring {document_id} as it already exists')
        return None

    if os.path.exists(video_name):
        print(f"Video {video_name} already exists")

    if not os.path.exists(video_path):
        print(f"Video {video_name} not exists - downloading")
        os.system(f"curl --user {username}:{password} {video_url} --output {video_path}")

    if len(glob.glob("tmp/*.json")) == 0:
        print(f"Extracting video from {video_name}")
        os.system(f"tar -xf {video_path} -C tmp/")
        return video_path

    return None

def get_keypoints(video_path):
    # Load keypoints
    keypoints = glob.glob("tmp/*.json")
    sorted_keypoints = sorted(keypoints)

    KEYPOINTS = []
    for keypoint in tqdm(sorted_keypoints):
        with open(keypoint, "r") as f:
            keypoints = json.load(f)
            if keypoints['people']:
                for person in keypoints['people']:
                    # Remove confidence scores

                    KEYPOINTS.append({
                        'person_id': person['person_id'][0],
                        'document_id': keypoint.split("/")[-1].replace(".json", "").split('_')[0],
                        'frame_id': keypoint.split("/")[-1].replace(".json", "").split('_')[1],
                        
                        'pose_keypoints_2d': np.array(
                            person['pose_keypoints_2d']
                        ).reshape(-1, 3)[:, :2].reshape(-1, 1),

                        'face_keypoints_2d': np.array(
                            person['face_keypoints_2d']
                        ).reshape(-1, 3)[:, :2].reshape(-1, 1),

                        'hand_left_keypoints_2d': np.array(
                            person['hand_left_keypoints_2d']
                        ).reshape(-1, 3)[:, :2].reshape(-1, 1),
                        
                        'hand_right_keypoints_2d': np.array(
                            person['hand_right_keypoints_2d']
                        ).reshape(-1, 3)[:, :2].reshape(-1, 1),
                    })

    return pd.DataFrame(KEYPOINTS)

def process_keypoints(df_keypoints, save_path="features"):
    os.makedirs(save_path, exist_ok=True)

    df_keypoints['keypoints'] = df_keypoints.apply(merge_keypoints, axis=1)

    array = np.array(df_keypoints.keypoints.to_list())

    with open(f"{save_path}/{df_keypoints.document_id.iloc[0]}.pkl", "wb") as f:
        pickle.dump(array, f)

    return {
        'document_id': df_keypoints.document_id.iloc[0],
        'frame_ids': df_keypoints.frame_id.to_list(),
        'person_count': len(df_keypoints.person_id.unique()),
        'left_hand_missing': int((df_keypoints.hand_left_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()),
        'right_hand_missing': int((df_keypoints.hand_right_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()),
        'face_missing': int((df_keypoints.face_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()),
        'pose_missing': int((df_keypoints.pose_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()),

    }

def remove_files():
    os.system("rm -rf tmp")


def main():

    # # Load dataset urls
    with open("config/bobsl.txt", "r") as f:
        dataset = f.readlines()

    dataset = [video_url.replace('\n', '') for video_url in dataset if video_url.startswith("http")]

    config = {
        "username": os.getenv("BOBSL_USERNAME"),
        "password": os.getenv("BOBSL_PASSWORD"),
    }

    if config["username"] is None or config["password"] is None:
        raise Exception("Please set BOBSL_USERNAME and BOBSL_PASSWORD environment variables")

    dataset_info = []
    for video_url in dataset[:10]:
        video_path = download_video(video_url, config["username"], config["password"])
        if video_path is None:
            print(f"Skipping {video_url}")
            continue
        
        df_keypoints = get_keypoints(video_path)
        info = process_keypoints(df_keypoints)
        dataset_info.append(info)

        with open("info.json", "w") as f: json.dump(dataset_info, f)

        remove_files()
        

if __name__ == "__main__":
    main()