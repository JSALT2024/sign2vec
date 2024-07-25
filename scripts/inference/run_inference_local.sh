# rm -rf pretraining/how2sign/sign2vec.dev.0.h5

python3 sign2vec/utils/run_sign2vec_inference_tti.py --model_name karahansahin/sign2vec-yasl-sc-sc-64-8-d1 \
                                                 --data_dir pretraining/how2sign/ \
                                                 --output_path pretraining/how2sign/tti \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file sign2vec/how2sign/h2s.annotations.dev.json
