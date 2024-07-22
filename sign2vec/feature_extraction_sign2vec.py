# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extractor class for Sign2Vec
"""

from typing import List, Optional, Union

import torch
import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import PaddingStrategy, TensorType, logging
from .utils.normalization import local_keypoint_normalization, global_keypoint_normalization
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor

logger = logging.get_logger(__name__)


class Sign2VecFeatureExtractor(Wav2Vec2FeatureExtractor):
    r"""
    Constructs a Sign2Vec feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which pose landmarks were sampled. Defaults to 25fps.
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*,
            [Sign2Vec-lv60](https://huggingface.co/models?search=lv60).
        max_duration_in_seconds (`float`, *optional*, defaults to 20.0):
            The maximum duration in seconds of the input sequence. Longer sequences will be truncated.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~Sign2VecFeatureExtractor.__call__`] should return `attention_mask`.
        kp_norm (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize the keypoints from with methodology from Spoter2 \cite{Bohacek_2022_WACV}.
            <Tip>

            </Tip>"""

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=25,
        padding_value=0.0,
        max_duration_in_seconds=30.0,
        return_attention_mask=False,
        kp_norm=False,
        do_normalize=True,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, max_duration_in_seconds=max_duration_in_seconds, return_attention_mask=return_attention_mask, do_normalize=do_normalize)
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            do_normalize=self.do_normalize,
        )

        self.max_duration_in_seconds = max_duration_in_seconds
        self.padding_value = padding_value

        self.kp_norm = kp_norm
        self.max_length = int(max_duration_in_seconds * sampling_rate)

        self.face_landmarks = [
            0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81, 
            93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
            282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
        ]
        self.pose_landmarks = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 ]

        self.kp_norm = kp_norm
        self.norm = [
            "global-pose_landmarks",
            "local-right_hand_landmarks", # "local-right_hand_landmarks",
            "local-left_hand_landmarks", # "local-left_hand_landmarks",
            "local-face_landmarks"
        ]


    def transform_keypoints(self, landmarks):
        """
        Transform the keypoints into a format that can be used by the model.
        
        Args:
            keypoints (dict): The keypoints to be transformed.
            """

        if self.norm:
            local_landmarks = {}
            global_landmarks = {}

            for idx, landmarks in enumerate(self.norm):
                prefix, landmarks = landmarks.split("-")
                if prefix == "local":
                    local_landmarks[idx] = landmarks
                elif prefix == "global":
                    global_landmarks[idx] = landmarks

            # local normalization
            for idx, landmarks in local_landmarks.items():
                normalized_keypoints = local_keypoint_normalization(landmarks, landmarks, padding=0.2)
                local_landmarks[idx] = normalized_keypoints

            # global normalization
            additional_landmarks = list(global_landmarks.values())
            if "pose_landmarks" in additional_landmarks:
                additional_landmarks.remove("pose_landmarks")
            keypoints, additional_keypoints = global_keypoint_normalization(
                landmarks,
                "pose_landmarks",
                additional_landmarks
            )
            for k,  landmark in global_landmarks.items():
                if landmark == "pose_landmarks":
                    global_landmarks[k] = keypoints
                else:
                    global_landmarks[k] = additional_keypoints[landmark]

            all_landmarks = {**local_landmarks, **global_landmarks}
            data = []
            for idx in range(len(self.norm)):
                data.append(all_landmarks[idx])

            pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = data

        # select only wanted KPI and  x, y coordinates
        pose_landmarks = pose_landmarks[:, self.pose_landmarks, :2]
        right_hand_landmarks = right_hand_landmarks[:, :, :2]
        left_hand_landmarks = left_hand_landmarks[:, :, :2]
        face_landmarks = face_landmarks[:, self.face_landmarks, :2]  

        pose_landmarks = pose_landmarks.reshape( pose_landmarks.shape[0], -1 )
        right_hand_landmarks = right_hand_landmarks.reshape( right_hand_landmarks.shape[0], -1 )
        left_hand_landmarks = left_hand_landmarks.reshape( left_hand_landmarks.shape[0], -1 )
        face_landmarks = face_landmarks.reshape( face_landmarks.shape[0], -1 )

        data = np.concatenate([pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks], axis=1)

        data = torch.tensor(data).reshape(data.shape[0], -1)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return data

    def __call__(
        self,
        pose_landmarks: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                Sign2Vec models that have set `config.feat_extract_norm == "group"`, such as
                [Sign2Vec-base](https://huggingface.co/facebook/Sign2Vec-base-960h), have **not** been trained using
                `attention_mask`. For such models, `input_values` should simply be padded with 0 and no
                `attention_mask` should be passed.

                For Sign2Vec models that have set `config.feat_extract_norm == "layer"`, such as
                [Sign2Vec-lv60](https://huggingface.co/facebook/Sign2Vec-large-960h-lv60-self), `attention_mask` should
                be passed for batched inference.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding_value (`float`, defaults to 0.0):
        """

        if isinstance(pose_landmarks, list):
            videos = [ self.transform_keypoints(landmarks) for landmarks in pose_landmarks ]
        if isinstance(pose_landmarks, dict):
            videos = [ self.transform_keypoints(pose_landmarks) ]

        processed_landmarks = self.feature_extractor(
            videos, 
            max_length=self.max_length, 
            truncation=True, 
            sampling_rate=25,
        )

        return processed_landmarks
