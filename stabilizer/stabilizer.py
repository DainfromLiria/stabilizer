"""
    Module that stabilizes the input video in real time.
"""
import depthai as dai
import numpy as np
import cv2
from typing import Tuple, Any

SMOOTHING_RADIUS = 50


class Stabilizer:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        # camera initialization
        self.cam = self.pipeline.createColorCamera()
        self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam.setPreviewSize(600, 600)
        self.cam.setInterleaved(False)
        # output video stream
        xout_video = self.pipeline.createXLinkOut()
        xout_video.setStreamName("video")
        # link plugins IMU -> XLINK and Camera -> XLINK
        self.cam.preview.link(xout_video.input)  # CHANGE preview to video

        self.transform_matrix = np.eye(3, dtype=np.float32)
        self.trajectory = []
        self.smoothed_trajectory = []
        self.prev_frame = None
        self.prev_pts = None

    def run(self):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
            video_queue = device.getOutputQueue(name="video")

            while True:
                in_video = video_queue.tryGet()  # Get image frame
                if in_video is not None:
                    frame = in_video.getCvFrame()

                    s_frame = self.stabilize_frame(frame)


                    cv2.putText(frame, 'Before', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)  # Add text to the original frame
                    cv2.putText(s_frame, 'After', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)  # Add text to the original frame
                    combined_frame = np.hstack((frame, s_frame))  # combine before and after frames
                    cv2.imshow("video", combined_frame)  # show the frame

                # Exit on esc or q
                if cv2.waitKey(1) in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()

    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
        dx, dy, da = self.estimate_motion(gray)

    def estimate_motion(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect feature points in the previous and current frame.
        Estimate the motion between previous and current frame.

        Return:
            dx - translation vector on x-axis.
            dy - translation vector on y-axis.
            da - rotation vector on x-axis.
        """
        prev_pts = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=200,
                                           qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, prev_pts, nextPts=None)

        # Filter only valid points
        prev_pts = prev_pts[status == 1]
        curr_pts = curr_pts[status == 1]

        transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        return dx, dy, da
