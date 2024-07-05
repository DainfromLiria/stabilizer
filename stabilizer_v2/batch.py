"""
    Class, that represent one batch of data from camera data queue.
    Also contains smoothed rotation vector (quaternions) and stabilized frame.
"""
# image processing
import numpy as np
import cv2
from quaternion import quaternion, as_rotation_matrix, slerp
# package moduls
from stabilizer_v2.settings import CROP_FACTOR


class Batch:
    def __init__(self, frame: np.ndarray, quat: quaternion):
        self.orig_frame = frame
        self.orig_orient = quat
        self.smooth_orient = quat
        self.stab_frame = frame
        self.__w = frame.shape[1]
        self.__h = frame.shape[0]

    def smooth_orientation(self, begin: quaternion, end: quaternion):
        self.smooth_orient = slerp(begin, end, 1, 3, 1)

    def create_rotation_matrix(self) -> np.ndarray:
        """Create 2D rotation matrix from stabilized rotation vector.

        Return:
            np.ndarray: 3x3 rotation matrix.
        """
        orient_norm = self.smooth_orient.normalized()
        h = as_rotation_matrix(orient_norm)
        # transform matrix from 3D to 2D
        h[2][2] = 1.0
        h[2][0] = h[2][1] = h[0][2] = h[1][2] = 0.0
        print(h)
        return h

    def warp_frame(self, cf: float = CROP_FACTOR) -> None:
        """Warp frame using 2D rotation matrix created from stabilized rotation vector.

        Parameter:
            cp - float (between 0 and 1) - percentage of the original image
            in both width and height in stabilized frame.
        """
        h = self.create_rotation_matrix()
        # print(f"w: {self.__w}, h: {self.__h}")
        self.stab_frame = cv2.warpPerspective(self.orig_frame, h, (self.__w, self.__h))
        new_size = (int(cf * self.__w), int(cf * self.__h))
        self.stab_frame = cv2.resize(self.stab_frame, dsize=new_size)
