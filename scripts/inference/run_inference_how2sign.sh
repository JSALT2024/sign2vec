export OUTPUT_DIR=/ssd1/karahan/

mkdir -p ${OUTPUT_DIR}/How2Sign/

MODEL_NAME=sign2vec-yasl-mc-sc-lconv-128

LAYER=-1
mkdir -p ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}



export LAYER=3
mkdir -p ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}



export LAYER=6
mkdir -p ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}



export LAYER=9
mkdir -p ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json \
                      s                           --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json \
                                                 --layer_to_extract ${LAYER}

python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/${MODEL_NAME} \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/sign2vec/${MODEL_NAME}_layer=${LAYER} \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json\
                                                 --layer_to_extract ${LAYER}
