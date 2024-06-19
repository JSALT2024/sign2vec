import os
import glob
import json
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

POSE_COLS = [
    "pose_keypoints_2d",
    "face_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
]


def merge_keypoints(row):
    return np.concatenate([row[pose_col] for pose_col in POSE_COLS])


def download_video(video_url, username, password):

    if not video_url.startswith("http"):
        return None

    video_url = video_url.replace("\n", "")
    video_name = video_url.split("/")[-1].strip().replace("\n", "")
    video_path = os.path.join(os.path.dirname(__file__), f"tmp/{video_name}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "tmp/"), exist_ok=True)
    document_id = video_name.split("_")[0].strip().split(".")[0]

    print(f"Processing {document_id}")

    FEATURE_PATH = os.path.join(
        os.path.dirname(__file__), f"features/{document_id}.npy"
    )
    VIDEO_PATH = os.path.join(os.path.dirname(__file__), f"tmp/{video_name}")
    EXPORT_PATH = os.path.join(os.path.dirname(__file__), f"tmp/{document_id}_*.json")

    if os.path.exists(FEATURE_PATH):
        print(f"Ignoring {FEATURE_PATH} as it already exists")
        return None

    if os.path.exists(VIDEO_PATH):
        print(f"Video {VIDEO_PATH} already exists")
        if len(glob.glob(EXPORT_PATH)) == 0:
            print(f"Extracting video from {video_name}")
            os.system(f"tar -xf {video_path} -C {os.path.dirname(__file__)}/tmp/")
        else: print('Already extracted')
        return video_path

    if not os.path.exists(VIDEO_PATH):
        print(f"Video {VIDEO_PATH} not exists - downloading")
        os.system(
            f"curl --user {username}:{password} {video_url} --output {VIDEO_PATH}"
        )


    if len(glob.glob(EXPORT_PATH)) == 0:
        print(f"Extracting video from {video_name}")
        os.system(f"tar -xf {video_path} -C {os.path.dirname(__file__)}/tmp/")
        return video_path
    

    return None

def get_keypoints(video_path):
    # Load keypoints
    TMP_PATH = os.path.join(os.path.dirname(__file__), "tmp/*.json")

    keypoints = glob.glob(TMP_PATH)
    sorted_keypoints = sorted(keypoints)

    KEYPOINTS = []
    for keypoint in tqdm(sorted_keypoints):
        with open(keypoint, "r") as f:
            keypoints = json.load(f)
            if keypoints["people"]:
                for person in keypoints["people"]:
                    # Remove confidence scores

                    KEYPOINTS.append(
                        {
                            "person_id": person["person_id"][0],
                            "document_id": keypoint.split("/")[-1]
                            .replace(".json", "")
                            .split("_")[0],
                            "frame_id": keypoint.split("/")[-1]
                            .replace(".json", "")
                            .split("_")[1],
                            "pose_keypoints_2d": np.array(person["pose_keypoints_2d"])
                            .reshape(-1, 3)[:, :2] # Remove confidence scores
                            .reshape(-1, 1),
                            "face_keypoints_2d": np.array(person["face_keypoints_2d"])
                            .reshape(-1, 3)[:, :2] # Remove confidence scores
                            .reshape(-1, 1),
                            "hand_left_keypoints_2d": np.array(
                                person["hand_left_keypoints_2d"]
                            )
                            .reshape(-1, 3)[:, :2] # Remove confidence scores
                            .reshape(-1, 1),
                            "hand_right_keypoints_2d": np.array(
                                person["hand_right_keypoints_2d"]
                            )
                            .reshape(-1, 3)[:, :2] # Remove confidence scores
                            .reshape(-1, 1),
                        }
                    )

    return pd.DataFrame(KEYPOINTS)


def process_keypoints(df_keypoints, save_path="features"):
    
    document_id = df_keypoints.document_id.iloc[0]

    SAVE_PATH = os.path.join(
        os.path.dirname(__file__), f"{save_path}/{document_id}.npy"
    )

    os.makedirs(os.path.join(
        os.path.dirname(__file__), f"{save_path}"
    ), exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    def merge_keypoints(row):
        return np.concatenate([row[pose_col] for pose_col in POSE_COLS])

    df_keypoints["array"] = df_keypoints.apply(merge_keypoints, axis=1)

    array = np.concatenate(df_keypoints["array"].values)

    max_frame = sorted(glob.glob(os.path.join(os.path.dirname(__file__),  f"tmp/{document_id}_*.json")))[-1]
    
    max_frame = max_frame.split("/")[-1].replace(".json", "").split("_")[1]
    max_frame = int(max_frame)

    with open(SAVE_PATH, "wb") as f:
        np.save(f, array)

    return {
        "document_id": df_keypoints.document_id.iloc[0],
        "person_count": len(df_keypoints.person_id.unique()),
        "left_hand_missing": int(
            (df_keypoints.hand_left_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()
        ),
        "right_hand_missing": int(
            (df_keypoints.hand_right_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()
        ),
        "face_missing": int(
            (df_keypoints.face_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()
        ),
        "pose_missing": int(
            (df_keypoints.pose_keypoints_2d.apply(lambda x: x.sum()) == 0).sum()
        ),
        "total_frames": max_frame,
        "nan_frames": max_frame - len(df_keypoints),
        "frame_ids": df_keypoints.frame_id.to_list(),
    }


def remove_files():
    os.system("rm -rf tmp")


def main():

    import os

    SOURCE_PATH = os.path.join(os.path.dirname(__file__), "config/bobsl.txt")
    INFO_PATH = os.path.join(os.path.dirname(__file__), "config/info.json")

    # # Load dataset urls
    with open(SOURCE_PATH, "r") as f:
        dataset = f.readlines()

    dataset = [
        video_url.replace("\n", "")
        for video_url in dataset
        if video_url.startswith("http")
    ]

    config = {
        "username": os.getenv("BOBSL_USERNAME"),
        "password": os.getenv("BOBSL_PASSWORD"),
    }

    if config["username"] is None or config["password"] is None:
        raise Exception(
            "Please set BOBSL_USERNAME and BOBSL_PASSWORD environment variables"
        )

    dataset_info = []
    dataset = dataset[:100] if os.getenv("DEBUG") else dataset
    for video_url in dataset:
        video_path = download_video(video_url, config["username"], config["password"])

        if video_path is None:
            print(f"Skipping {video_url}")
            continue

        df_keypoints = get_keypoints(video_path)
        info = process_keypoints(df_keypoints)
        dataset_info.append(info)

        with open(INFO_PATH, "w") as f:
            json.dump(dataset_info, f)

        os.system(f'rm -rf {os.path.join(os.path.dirname(__file__), "tmp/")}')


if __name__ == "__main__":
    main()
