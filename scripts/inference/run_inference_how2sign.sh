export OUTPUT_DIR=sign2vec/data/how2sign

rm -rf ${OUTPUT_DIR}/How2Sign/sign2vec/sign2vec.train.0.h5
rm -rf ${OUTPUT_DIR}/How2Sign/sign2vec/sign2vec.dev.0.h5
rm -rf ${OUTPUT_DIR}/How2Sign/sign2vec/sign2vec.test.0.h5

mkdir -p ${OUTPUT_DIR}/How2Sign/

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file sign2vec/how2sign/H2S.annotations.train.json

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file sign2vec/how2sign/H2S.annotations.dev.json

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file sign2vec/how2sign/H2S.annotations.test.json

