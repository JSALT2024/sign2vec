import os
import h5py
import pandas as pd

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--pose_dir', type=str, default='../YASL_poses/')
    parser.add_argument('--output_path', type=str, default='/home/azureuser/cloudfiles/code/Users/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    files = os.listdir(args.pose_dir)
    df_keys = []
    for file in files:
        try:
            with h5py.File(os.path.join(args.pose_dir,file), 'r') as f:
                keys = list(f.keys())
                df_keys.extend([{
                    'video_id': key,
                    'sentence_idx': idx,
                    'h5_file_path': file,
                } for idx, key in enumerate(keys)])
        except Exception as e:
            print(e)

    df_keys = pd.DataFrame.from_records(df_keys)

    from sklearn.model_selection import train_test_split

    train, val = train_test_split(df_keys, test_size=args.test_size, random_state=args.seed)
    train.to_csv(os.path.join(args.output_path,'train_dataset.csv'), index=False)
    val.to_csv(os.path.join(args.output_path,'val_dataset.csv'), index=False)