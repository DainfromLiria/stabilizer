"""
    Main module of package. Provide real-time video stabilization using
    data from IMU (rotation vector in quaternions) and Camera (video frames).
"""
# data IO
import depthai as dai
# data processing
import cv2
from datetime import timedelta
from collections import deque
from quaternion import quaternion, slerp, as_rotation_matrix
# package moduls
from stabilizer_v2.settings import BUFFER_SIZE
from stabilizer_v2.batch import Batch


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

        # stabilization
        self.__buffer: deque[Batch] = deque()

    def receive_data(self, group_queue: dai.DataOutputQueue) -> None:
        """
        Receive one frame and rotation vector (in quaternions) from synchronize data queue.
        Received data (including timestamp) is stored in the buffer as Batch objects.
        """
        group_mess = group_queue.get()
        imu_mess = group_mess["imu"]
        cam_mess = group_mess["video"]
        timestamp = imu_mess.getTimestampDevice()  # TODO maybe delete
        frame = cam_mess.getCvFrame()
        quat = imu_mess.packets[-1].rotationVector
        print("===========================================")
        print("Device timestamp imu: " + str(imu_mess.getTimestampDevice()))
        print("Device timestamp video: " + str(cam_mess.getTimestampDevice()))
        imuF = "{:.4f}"
        print(f"Quaternion: i: {imuF.format(quat.i)} j: {imuF.format(quat.j)} "
              f"k: {imuF.format(quat.k)} real: {imuF.format(quat.real)}")
        print("===========================================")
        self.__buffer.append(Batch(frame, quaternion(quat.real, quat.i, quat.j, quat.k), timestamp))

    def run(self) -> None:
        """Run real-time stabilization."""
        with dai.Device(self.__pipeline) as device:
            group_queue = device.getOutputQueue("xout", 3, True)
            # Fill buffer by initial frames
            for _ in range(BUFFER_SIZE):
                self.receive_data(group_queue)
            # Stabilize next frames one by one
            while True:
                self.receive_data(group_queue)  # read last frame for full buffer
                self.stabilize()
                cv2.imshow("video", self.__buffer[0].stab_frame)
                self.__buffer.popleft()  # pop first
                # Exit on esc or q
                if cv2.waitKey(1) in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()

    def stabilize(self):
        curr = self.__buffer[1].smooth_orient
        # curr = slerp(self.__buffer[0].smooth_orient, self.__buffer[2].smooth_orient, 1, 3, 2)
        h = as_rotation_matrix(curr)
        self.__buffer[1].stab_frame = cv2.warpPerspective(self.__buffer[1].orig_frame, h, (1920, 1080))
