def update_sentence_at_h5(h5_fpath, csv_fpath):
    import h5py
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(csv_fpath, sep='\t')
    
    with h5py.File(h5_fpath, 'r+') as f:
        for i, row in tqdm(df.iterrows()):
            try:
                del f[f'{row.SENTENCE_NAME}/sentence']
                f.create_dataset(f'{row.SENTENCE_NAME}/sentence', data=row.SENTENCE)

            except Exception as e:
                print(f'Error at {row.SENTENCE_NAME}: {e}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Update sentence at H5 file')
    parser.add_argument('--h5_fpath', type=str, help='H5 file path')
    parser.add_argument('--csv_fpath', type=str, help='CSV file path')
    return parser.parse_args()

if __name__ == '__main__':
    import os
    
    args = parse_args()

    update_sentence_at_h5(
        h5_fpath=os.path.join(args.h5_fpath, 'H2S_train.h5'),
        csv_fpath=os.path.join(args.csv_fpath, 'how2sign_realigned_train.csv')
    )

    update_sentence_at_h5(
        h5_fpath=os.path.join(args.h5_fpath, 'H2S_val.h5'),
        csv_fpath=os.path.join(args.csv_fpath, 'how2sign_realigned_val.csv')
    )

    update_sentence_at_h5(
        h5_fpath=os.path.join(args.h5_fpath, 'H2S_test.h5'),
        csv_fpath=os.path.join(args.csv_fpath, 'how2sign_realigned_test.csv')
    )