import math
import numpy as np


def normalize_local(keypoints, keypoint_type):
    raise NotImplementedError('Local normalization is not implemented yet')

def normalize_global(keypoints, keypoint_type):
    raise NotImplementedError('Global normalization is not implemented yet')

def normalize_yasl(landmarks):
    """
    Normalize the keypoints of the YASL dataset.

    Args:
        keypoints (List(np.ndarray)): The keypoints to normalize.
        keypoint_type (str): The type of keypoints to normalize.

    Returns:
        np.ndarray: The normalized keypoints.
    """
    # represent landmarks that are not present in a frame with a large negative value
    # this is necessary for the normalization to work correctly

    # if one row is all zeros, the frame is empty -> set all values to -1
    landmarks[landmarks[:,:,:2].sum(axis=2) == 0] = -1
    
    min_x_y = np.min(landmarks[:, :, :2], axis=1)
    max_x_y = np.max(landmarks[:, :, :2], axis=1)

    landmarks[:,:,:2] = (landmarks[:,:,:2] - min_x_y[:, None, :]) / (
        max_x_y - min_x_y
    )[:, None, :]
    # Normalize the keypoints
    return landmarks