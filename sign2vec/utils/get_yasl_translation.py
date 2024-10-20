train_annot = json.load(open('/ssd2/karahan/YASL/annotations/train.json'))
dev_annot = json.load(open('/ssd2/karahan/YASL/annotations/dev.json'))

train_df = pd.read_csv('/ssd2/karahan/YASL/pose/yasl_train.csv')
val_df = pd.read_csv('/ssd2/karahan/YASL/pose/yasl_val.csv')

def get_sentence(clip_id): 
    video_id, _ = clip_id.split('.')
    try:
            if train_annot.get(video_id):
                    return train_annot.get(video_id).get(clip_id).get('translation')
    except:
           print(f"Clip ID {clip_id} not found in the train dataset")
           pass
    try:
            if dev_annot.get(video_id):
                    return dev_annot.get(video_id).get(clip_id).get('translation')
    except:
            print(f"Clip ID {clip_id} not found in the dev dataset")
            pass
            
    print(f"Clip ID {clip_id} not found in the datasets")
    print('*'*50)
    return ''

train_df['sentence'] = train_df.clip_id.progress_apply(get_sentence)
val_df['sentence'] = val_df.clip_id.progress_apply(get_sentence)
test_df['sentence'] = test_df.clip_id.progress_apply(get_sentence)

train_df = pd.read_csv('/ssd2/karahan/YASL/pose/yasl_train.csv')
val_df = pd.read_csv('/ssd2/karahan/YASL/pose/yasl_val.csv')
test_df = pd.read_csv('/ssd2/karahan/YASL/pose/yasl_test.csv')

def process_sentence(sentence):
    if sentence == '':
        return 'No translation'
    elif isinstance(sentence, str):
        return sentence
    elif isinstance(sentence, list):
        return ' '.join(sentence)
    else:
        return 'No translation'

train_df['sentence'] = train_df.sentence.apply(process_sentence)
val_df['sentence'] = val_df.sentence.apply(process_sentence)
test_df['sentence'] = test_df.sentence.apply(process_sentence)

train_df.to_csv('/ssd2/karahan/YASL/pose/yasl_train.csv', index=False)
val_df.to_csv('/ssd2/karahan/YASL/pose/yasl_val.csv', index=False)
test_df.to_csv('/ssd2/karahan/YASL/pose/yasl_test.csv', index=False)