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
from quaternion import quaternion
# package moduls
from stabilizer_v2.settings import BUFFER_SIZE, OFFSET_UPDATE_DELAY, CURRENT_FRAME_IDX
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
        self.__offset = None
        self.__delay = 0

    def receive_data(self, group_queue: dai.DataOutputQueue) -> None:
        """
        Receive one frame and rotation vector (in quaternions) from synchronize data queue.
        Received data (including timestamp) is stored in the buffer as Batch objects.

        Parameter:
            group_queue - dai.DataOutputQueue - data queue with data from camera and imu.
        """
        group_mess = group_queue.get()
        imu_mess = group_mess["imu"]
        cam_mess = group_mess["video"]
        frame = cam_mess.getCvFrame()
        quat = imu_mess.packets[-1].rotationVector
        print("===========================================")
        imuF = "{:.4f}"
        print(f"Quaternion: i: {imuF.format(quat.i)} j: {imuF.format(quat.j)} "
              f"k: {imuF.format(quat.k)} real: {imuF.format(quat.real)}")
        if self.__offset is None or self.__delay >= OFFSET_UPDATE_DELAY:
            self.__offset = quaternion(1, 0, 0, 0) - quaternion(quat.real, quat.i, quat.j, quat.k)
            self.__delay = 0
        else:
            self.__delay += 1
        self.__buffer.append(Batch(frame, quaternion(quat.real, quat.i, quat.j, quat.k) + self.__offset))

    def run(self) -> None:
        """Run real-time video stabilization."""
        with dai.Device(self.__pipeline) as device:
            group_queue = device.getOutputQueue("xout", 3, True)
            # Fill buffer by initial frames
            for _ in range(BUFFER_SIZE):
                self.receive_data(group_queue)
            # Stabilize next frames one by one
            while True:
                self.receive_data(group_queue)  # read last frame for full buffer
                self.stabilize()
                cv2.imshow("stab", self.__buffer[0].stab_frame)
                cv2.imshow("orig", self.__buffer[0].orig_frame)
                self.__buffer.popleft()  # pop first
                # Exit on esc or q
                if cv2.waitKey(1) in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()

    def stabilize(self) -> None:
        """Stabilize one frame from video."""
        begin = self.__buffer[CURRENT_FRAME_IDX].smooth_orient
        end = self.__buffer[BUFFER_SIZE].smooth_orient
        self.__buffer[CURRENT_FRAME_IDX].smooth_orientation(begin, end)
        print(f"curr: {self.__buffer[CURRENT_FRAME_IDX].smooth_orient}")
        print(f"self.offset: {self.__offset}")
        self.__buffer[CURRENT_FRAME_IDX].warp_frame()
        print("===========================================")
