import numpy as np
import tflite_runtime.interpreter as tflite
# import tensorflow.lite as tflite
import cv2
import time
import os
import collections
import math
import enum

class PoseRecognition:
    def __init__(self,
                 tflite_file: str = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'):
        # Загрузка нейроной сети из tflite файла
        self.interpreter = tflite.Interpreter(model_path=tflite_file,
            experimental_delegates=[
                tflite.load_delegate('libedgetpu.so.1'),                
                tflite.load_delegate(os.path.join('./', './', 'posenet_decoder.so'))
            ])
        self.interpreter.allocate_tensors()
        # Получение информации о входном и выходном слое
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # Размеры входного слоя == обрабатываемого изображения
        self.input_image_shape = self.input_details[0]['shape'][1:]
        self.input_image_shape = np.array([
            self.input_image_shape[1],
            self.input_image_shape[0],
            self.input_image_shape[2]
        ])

        self.time_net = []
        self.time_img = []
        self.time_parse=[]

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: Обрабатываемое изображение.
        :return: Коодринаты ключевых точек, частей тела.
        """
        start = time.time()

        image = self.__resize_image(image)
        image = self.__image_prepare(image)

        start_net = time.time()
        output_heatmap, output_offset = self.__set_and_run_interpreter(image=image)
        stop_net = time.time()
    
        target_heatmaps = np.squeeze(output_heatmap)
        target_offsets = np.squeeze(output_offset)

        # image = self.__image_unprepare(image)

        # target_kps = self.parse_output(target_heatmaps, target_offsets, 0.3)
        # target_kps = self.ParseOutput()
        # if len(target_kps) > 1:
        #     target_kps = self.ParsePose(target_kps[0])
        # else:
        #     target_kps = np.zeros((25, 3))  # fix later
        # target_kps = self.__resize_kps_back(target_kps)

        target_kps = [
            self.__resize_kps_back(
                self.ParsePose(
                    kps)) for kps in self.ParseOutput()
            ]

        stop_parse = time.time()

        self.time_net.append(stop_net - start_net)
        self.time_img.append(start_net - start)
        self.time_parse.append(stop_parse - stop_net)

        return target_kps

    @staticmethod
    def __image_prepare(image: np.ndarray) -> np.ndarray:
        return image
        return (image.astype(np.float32) - 127.5) / 127.5

    @staticmethod
    def __image_unprepare(image: np.ndarray) -> np.ndarray:
        return (image * 127.5 + 127.5).astype(np.uint8)

    def __resize_image(self, image: np.ndarray) -> np.ndarray:
        # Скейлинг изображения к размеру подходящем для входа в нейросеть
        self.original_shape = image.shape
        if all(map(lambda x: x[0] == x[1], zip(self.input_image_shape, self.original_shape))):
            return image
        # TODO: единственное место где opencv используется, потенциально можно избавиться от зависимости
        return cv2.resize(image, tuple(self.input_image_shape[:2]))

    def __resize_kps_back(self, kps: np.ndarray) -> np.ndarray:
        # Скейлниг координат точек обратно к размеру исходного изображения
        if all(map(lambda x: x[0] == x[1], zip(self.input_image_shape, self.original_shape))):
            return kps
        for i in range(0, kps.shape[0]):
            # kps[i, 0] = 1.33 * kps[i, 0] * self.original_shape[0] / self.input_image_shape[0]
            # kps[i, 1] = 0.76 * kps[i, 1] * self.original_shape[1] / self.input_image_shape[1]
            kps[i, 0] = kps[i, 0] * self.original_shape[0] / self.input_image_shape[1]
            kps[i, 1] = kps[i, 1] * self.original_shape[1] / self.input_image_shape[0]
        return kps

    def __set_and_run_interpreter(self, image: np.ndarray) -> tuple:
        # Установить изображение в качетсве входа в нейронную сеть
        self.interpreter.set_tensor(
            tensor_index=self.input_details[0]['index'],
            value=[image]
        )
        # Запустить  исполение нейросети
        self.interpreter.invoke()

        output_heatmap = self.interpreter.get_tensor(
            tensor_index=self.output_details[0]['index']
        )
        output_offset = self.interpreter.get_tensor(
            tensor_index=self.output_details[1]['index']
        )
        return output_heatmap, output_offset

    @staticmethod
    def parse_output(heatmap_data: np.ndarray,
                     offset_data: np.ndarray,
                     threshold: float) -> np.ndarray:
        """
        :param heatmap_data: Карта вероятностей, 3d массив.
        :param offset_data: Коорднаты, 3d массив.
        :param threshold: Минимальная вероятность (0...1).
        :return: Координаты ключевых точек человека, помеченые с малой вероятностью.
        """

        joint_num = heatmap_data.shape[-1]
        pose_kps = np.zeros((joint_num, 3), np.uint32)

        for i in range(heatmap_data.shape[-1]):
            joint_heatmap = heatmap_data[..., i]
            max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
            remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)
            pose_kps[i, 0] = int(remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], i])
            pose_kps[i, 1] = int(remap_pos[1] + offset_data[max_val_pos[0], max_val_pos[1], i + joint_num])
            max_prob = np.max(joint_heatmap)

            if max_prob > threshold:
                if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                    pose_kps[i, 2] = 1

        return pose_kps

    def get_output_tensor(self, idx):
        """Returns output tensor view."""
        return np.squeeze(self.interpreter.tensor(
            self.interpreter.get_output_details()[idx]['index'])())

    def ParseOutput(self):
        """Parses interpreter output tensors and returns decoded poses."""
        keypoints = self.get_output_tensor(0)
        keypoint_scores = self.get_output_tensor(1)
        pose_scores = self.get_output_tensor(2)
        num_poses = self.get_output_tensor(3)
        poses = []
        for i in range(int(num_poses)):
            pose_score = pose_scores[i]
            pose_keypoints = {}
            for j, point in enumerate(keypoints[i]):
                y, x = point
                # if self._mirror:
                #     y = self._input_width - y
                pose_keypoints[KeypointType(j)] = Keypoint(Point(x, y), keypoint_scores[i, j])
            poses.append(Pose(pose_keypoints, pose_score))
        return poses
        # return poses, self._inf_time

    @staticmethod
    def ParsePose(pose):
        r = np.zeros((17, 3))
        for i in range(0, 17):
            x = pose.keypoints[i].point.x
            y = pose.keypoints[i].point.y
            s = pose.keypoints[i].score
            r[i][0] = y
            r[i][1] = x
            r[i][2] = s
        return r


Point = collections.namedtuple('Point', ['x', 'y'])
Point.distance = lambda a, b: math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
# Point.distance = staticmethod(Point.distance)

Keypoint = collections.namedtuple('Keypoint', ['point', 'score'])

Pose = collections.namedtuple('Pose', ['keypoints', 'score'])

class KeypointType(enum.IntEnum):
    """Pose kepoints."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
