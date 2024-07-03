import depthai as dai
import numpy as np
import cv2
from datetime import timedelta


class Stabilizer:
    def __init__(self):
        self.__pipeline = dai.Pipeline()
        # nodes
        cam = self.__pipeline.create(dai.node.ColorCamera)
        imu = self.__pipeline.create(dai.node.IMU)
        sync = self.__pipeline.create(dai.node.Sync)

        # XLinkOut
        xoutImu = self.__pipeline.create(dai.node.XLinkOut)
        xoutImu.setStreamName("imu")
        xoutGrp = self.__pipeline.create(dai.node.XLinkOut)
        xoutGrp.setStreamName("xout")

        # imu
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 120)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        # sync
        sync.setSyncThreshold(timedelta(milliseconds=10))
        sync.setSyncAttempts(-1)

        # link
        cam.video.link(sync.inputs["video"])
        imu.out.link(sync.inputs["imu"])
        sync.out.link(xoutGrp.input)

    def run(self):
        with dai.Device(self.__pipeline) as device:
            groupQueue = device.getOutputQueue("xout", 3, True)
            while True:
                groupMessage = groupQueue.get()
                imuMessage = groupMessage["imu"]
                colorMessage = groupMessage["video"]
                print()
                print("Device timestamp imu: " + str(imuMessage.getTimestampDevice()))
                print("Device timestamp video:" + str(colorMessage.getTimestampDevice()))
                latestRotationVector = imuMessage.packets[-1].rotationVector
                imuF = "{:.4f}"
                print(f"Quaternion: i: {imuF.format(latestRotationVector.i)} j: {imuF.format(latestRotationVector.j)} "
                      f"k: {imuF.format(latestRotationVector.k)} real: {imuF.format(latestRotationVector.real)}")
                print()
                cv2.imshow("video", colorMessage.getCvFrame())
                # Exit on esc or q
                if cv2.waitKey(1) in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()
