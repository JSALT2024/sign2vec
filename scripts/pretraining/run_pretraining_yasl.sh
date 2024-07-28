TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/royal/youtube_asl_mc_sc_lconv.yaml" 

TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/royal/youtube_asl_sc_sc_lconv.yaml" 
