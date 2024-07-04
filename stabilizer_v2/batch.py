"""
    Class, that represent one batch of data from camera data queue.
    Also contains smoothed rotation vector (quaternions) and stabilized frame.
"""
import numpy as np
from quaternion import quaternion


class Batch:
    def __init__(self, frame: np.ndarray, quat: quaternion, timestamp):
        self.orig_frame = frame
        self.orig_orient = quat
        self.smooth_orient = quat
        self.stab_frame = frame
        self.timestamp = timestamp  # TODO
