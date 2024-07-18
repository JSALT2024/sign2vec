python3 sign2vec/utils/run_sign2vec_inference.py --dataset_file pretraining/how2sign/H2S_val.csv \
                                                 --model_name karahansahin/sign2vec-yasl-k10-c128-d0.5 \
                                                 --data_dir pretraining/how2sign \
                                                 --output_path pretraining/how2sign \
                                                 --input_file H2S_val.h5 \
                                                 --output_file H2S_sign2vec_val.h5 \
                                                 --annotation_file sign2vec/how2sign/h2s.annotations.dev.json

python3 sign2vec/utils/run_sign2vec_inference.py --dataset_file pretraining/how2sign/H2S_train.csv \
                                                 --model_name karahansahin/sign2vec-yasl-k10-c128-d0.5 \
                                                 --data_dir pretraining/how2sign \
                                                 --output_path pretraining/how2sign \
                                                 --input_file H2S_train.h5 \
                                                 --output_file H2S_sign2vec_train.h5 \
                                                 --annotation_file sign2vec/how2sign/h2s.annotations.train.json

python3 sign2vec/utils/run_sign2vec_inference.py --dataset_file pretraining/how2sign/H2S_test.csv \
                                                 --model_name karahansahin/sign2vec-yasl-k10-c128-d0.5 \
                                                 --data_dir pretraining/how2sign \
                                                 --output_path pretraining/how2sign \
                                                 --input_file H2S_test.h5 \
                                                 --output_file H2S_sign2vec_test.h5 \
                                                 --annotation_file sign2vec/how2sign/h2s.annotations.test.json

