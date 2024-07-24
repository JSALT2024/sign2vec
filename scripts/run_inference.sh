rm -rf /ssd1/karahan/sign2vec.train.0.h5
rm -rf /ssd1/karahan/sign2vec.dev.0.h5
rm -rf /ssd1/karahan/sign2vec.test.0.h5

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-sc-sc-64-8-d1 \
                                                 --data_dir /ssd1/karahan/ \
                                                 --output_path /ssd1/karahan/ \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file sign2vec/how2sign/h2s.annotations.train.json

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-sc-sc-64-8-d1 \
                                                 --data_dir /ssd1/karahan/ \
                                                 --output_path /ssd1/karahan/ \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file sign2vec/how2sign/h2s.annotations.dev.json

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-sc-sc-64-8-d1 \
                                                 --data_dir /ssd1/karahan/ \
                                                 --output_path /ssd1/karahan/ \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file sign2vec/how2sign/h2s.annotations.test.json

