export OUTPUT_DIR=/ssd1/karahan/

mkdir -p ${OUTPUT_DIR}/How2Sign/

export MODEL_NAME=sign2vec-yasl-mc-sc-lconv-128

export LAYER=-1

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}



export LAYER=3

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}



export LAYER=6

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}



export LAYER=9

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}
