"""
    Module that stabilizes the input video in real time.
"""
import depthai as dai
import numpy as np
import cv2


class Stabilizer:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        # camera initialization
        self.cam = self.pipeline.createColorCamera()
        self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam.setPreviewSize(600, 600)
        self.cam.setInterleaved(False)
        # gyroscope initialization
        self.imu = self.pipeline.createIMU()
        self.imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 400)
        self.imu.setBatchReportThreshold(1)
        self.imu.setMaxBatchReports(10)
        # output video stream
        xout_video = self.pipeline.createXLinkOut()
        xout_video.setStreamName("video")
        # output gyroscope data stream
        xout_imu = self.pipeline.createXLinkOut()
        xout_imu.setStreamName("imu")
        # link plugins IMU -> XLINK and Camera -> XLINK
        self.cam.preview.link(xout_video.input)  # CHANGE preview to video
        self.imu.out.link(xout_imu.input)

    def run(self):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
            video_queue = device.getOutputQueue(name="video", maxSize=8, blocking=False)
            imu_queue = device.getOutputQueue(name="imu", maxSize=8, blocking=False)

            while True:
                # Get image frame
                in_video = video_queue.tryGet()
                if in_video is not None:
                    frame = in_video.getCvFrame()
                    cv2.putText(frame, 'Before', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)  # Add text to the original frame
                    combined_frame = np.hstack((frame, frame))  # combine before and after frames
                    cv2.imshow("video", combined_frame)  # show the frame

                # Get IMU data
                in_imu = imu_queue.tryGet()
                if in_imu is not None:
                    gyro = in_imu.packets[0].gyroscope
                    timestamp = gyro.getTimestampDevice()
                    print(f"Gyroscope: x: {gyro.x}, y: {gyro.y}, z: {gyro.z}, timestamp: {timestamp}")

                # Exit on esc or q
                if cv2.waitKey(1) in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()
