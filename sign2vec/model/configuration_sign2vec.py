from transformers import (
    PretrainedConfig,
    Wav2Vec2Config
)

class Sign2VecConfig(Wav2Vec2Config):

    model_type = "sign2vec"

    """
    This is the configuration class on top of which the Sign2Vec model is built.
    
    It inherits from :class:`~transformers.Wav2Vec2Config` and adds the following attributes:
    - :obj:`enable_multicue`: a boolean indicating whether the model is trained with multiple cues.
    - :obj:`input_dim`: an integer indicating the dimensionality of the input features.
    - :obj:`fps`: an integer indicating the frames per second of the input video.
    - :obj:`do_normalize`: a boolean indicating whether the input features should be normalized.
    """
    def __init__(self, 
                 input_dim: int = 255,
                 enable_multicue: bool = False,
                 conv_dim=[512, 512, 512, 512],
                 conv_stride=[2, 1, 1, 1],
                 conv_kernel=[10, 3, 3, 3],
                 num_conv_feat_extract_layers=4,
                 num_linear_feat_extract_layers=2,
                 pose_range=(0, 20),
                 face_range=(20, 30),
                 left_hand_range=(30, 50),
                 right_hand_range=(50, 70),
                 fps=25,
                 do_normalize=False,
                 encoder_type="conv_layers",
                 perplexity_reduction='mean',
                 **kwargs):
        super().__init__(**kwargs)

        self.model_type = "sign2vec"
        self.input_dim = input_dim

        self.fps = fps
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.num_conv_feat_extract_layers = num_conv_feat_extract_layers
        self.num_linear_feat_extract_layers = num_linear_feat_extract_layers
        self.encoder_type = encoder_type

        # Feature extractor config
        self.do_normalize = do_normalize

        # Multi-cue config
        self.enable_multicue = enable_multicue
        self.pose_range = pose_range
        self.face_range = face_range
        self.left_hand_range = left_hand_range
        self.right_hand_range = right_hand_range

        # Perplexity reduction config
        self.perplexity_reduction = perplexity_reduction

    def load_from_yaml(self, yaml_file: str):
        """
        Loads the configuration from a yaml file.
        """
        import yaml
        with open(yaml_file, "r") as yaml_file:
            yaml_dict = yaml.safe_load(yaml_file)
        return Sign2VecConfig(**yaml_dict)