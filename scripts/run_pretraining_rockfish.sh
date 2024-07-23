TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/rockfish/youtube_asl_mc_mc.yaml" 

TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/rockfish/youtube_asl_mc_sc.yaml" 

TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/rockfish/youtube_asl_sc_sc.yaml" 
