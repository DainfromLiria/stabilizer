"""
    Module that stabilizes the input video in real time.
"""
# data IO
import depthai as dai
# data processing
import numpy as np
import numpy.linalg as lin
import cv2
from collections import deque
# package moduls
from stabilizer.settings import *


class Stabilizer:
    def __init__(self, preview_mode: bool = False):
        """
            Initialize the stabilizer

            Parameter:
                preview_mode - bool - `True` for running aplication in preview mode, where user can
                see original and stabilized video, but video has resolution PREVIEW_RESOLUTION defined in settings.py.
                `False` for running in basic mode where user can see only stabilized video in 1080p.
        """
        self.__pipeline = dai.Pipeline()
        # camera
        cam = self.__pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(60.0)
        self.__w = cam.getResolutionWidth()
        self.__h = cam.getResolutionHeight()
        self.__preview_mode = preview_mode

        # XLinkOut
        xout_video = self.__pipeline.createXLinkOut()
        xout_video.setStreamName("video")

        # stabilization
        self.__prev_frame = None
        self.__buffer: deque = deque([np.identity(3)])
        self.__bp_threshold = BP_THRESHOLD
        self.__skip_warp = BUFFER_SIZE

        # preview mode
        if self.__preview_mode is True:
            self.__w = PREVIEW_RESOLUTION[0]
            self.__h = PREVIEW_RESOLUTION[1]
            cam.setPreviewSize(self.__w, self.__h)
            cam.preview.link(xout_video.input)
            self.__bp_threshold = BP_THRESHOLD_VIEW
        else:
            cam.video.link(xout_video.input)

    def receive_frame(self, video_queue: dai.DataOutputQueue) -> np.ndarray:
        """Receive one frame from DataOutputQueue."""
        in_video = video_queue.get()
        if self.__prev_frame is None:
            self.__prev_frame = cv2.cvtColor(in_video.getCvFrame(), cv2.COLOR_BGR2GRAY)
        return in_video.getCvFrame()

    def run(self):
        """Run real-time video stabilization."""
        with dai.Device(self.__pipeline) as device:
            video_queue = device.getOutputQueue(name="video")
            while True:
                frame = self.receive_frame(video_queue)
                stab_frame = self.stabilize_frame(frame)
                self.show_result(frame, stab_frame)
                # Exit on esc or q
                if cv2.waitKey(1) in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()

    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
            Stabilize input frame. If warped frame has more than BP_THRESHOLD
            black frames, then return cropped original frame and update buffer,
            otherwise return warped frame. After long camera move, return cropped original frame
            for more smoothed transition between the original frames and warped frames.
        """
        self.estimate_motion(frame)
        t_frame = self.smooth_and_warp(frame)
        if self.black_pix_count(t_frame) < self.__bp_threshold:
            if self.__skip_warp != 0:
                self.__skip_warp -= 1
                return self.crop_frame(np.copy(frame))
            return t_frame

        self.__buffer.clear()
        self.__buffer.append(np.identity(3))
        self.__skip_warp = BUFFER_SIZE
        return self.crop_frame(np.copy(frame))

    def black_pix_count(self, frame: np.ndarray) -> float:
        """
            Calculate count of black pixels in input image in percents.

            Returns:
                  float - from 0 to 100
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bp_count = np.sum(gray == 0)
        in_percent = (bp_count / (self.__h * self.__w)) * 100
        return in_percent

    def estimate_motion(self, frame: np.ndarray) -> None:
        """
            Detect feature points in the current frame. Using this points,
            calculate optical flow between current and previous frames (find this points
            in the previous frame). Using points from previous frames and current frame
            find homography 3x3 rotation matrix. Compute accumulate rotation matrix for this frame
            and add it on the end of the buffer.

            Returns:
                bool - True if distance between current and previous frames is
                in interval (MIN_DISTANCE, MAX_DISTANCE), otherwise False.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_pts = cv2.goodFeaturesToTrack(gray, **FEATURE_PARAMS)
        if curr_pts is not None:
            prev_pts, _, _ = cv2.calcOpticalFlowPyrLK(self.__prev_frame, gray, curr_pts, nextPts=None)
            if prev_pts.shape[0] >= 4 and prev_pts.shape[0] >= 4:
                h, _ = cv2.findHomography(curr_pts, prev_pts, cv2.RANSAC, 5.0)
                if h is not None:
                    self.__buffer.append(np.dot(h, self.__buffer[-1]))  # accumulate rotation matrix

        if len(self.__buffer) > BUFFER_SIZE:
            self.__buffer.popleft()
            # normalize buffer matrices by last matrix to
            # prevent errors after low or high frequency big shakes
            inv = lin.inv(self.__buffer[0])
            for i in range(len(self.__buffer)):
                self.__buffer[i] = np.dot(self.__buffer[i], inv)

        self.__prev_frame = gray

    def smooth_and_warp(self, frame: np.ndarray) -> np.ndarray:
        """
            Calculate average rotation matrix through all matrices in buffer and normalise it.
            Using this matrix, warp input image and crop it. In case if original matrix
            is singular (inversion does not exist) returns cropped version of original image.

            Parameter:
                frame - np.ndarray - original image.
            Return:
                np.ndarray - stabilized version of original image.
        """
        try:
            s_h = np.dot(np.average(np.array(self.__buffer), 0), lin.inv(self.__buffer[-1]))
        except lin.LinAlgError:
            return self.crop_frame(frame)
        s_frame = cv2.warpPerspective(frame, s_h, (self.__w, self.__h))
        return self.crop_frame(s_frame)

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
            Crop input frame on CROP_MARGIN percent, which is defined in settings.py
        """
        x_start = int(CROP_MARGIN * self.__w)
        y_start = int(CROP_MARGIN * self.__h)
        x_end = self.__w - x_start
        y_end = self.__h - y_start
        return frame[y_start:y_end, x_start:x_end]

    def show_result(self, orig_img: np.ndarray, stab_img: np.ndarray) -> None:
        """
            Show original image and stabilized image if preview mode is enabled.
            Otherwise, show only stabilized image.
        """
        if self.__preview_mode is True:
            cv2.putText(orig_img, 'Original', **TEXT_PARAMS)
            cv2.putText(stab_img, 'Stabilized', **TEXT_PARAMS)
            cv2.imshow("original", orig_img)
        cv2.imshow("stabilized", stab_img)
