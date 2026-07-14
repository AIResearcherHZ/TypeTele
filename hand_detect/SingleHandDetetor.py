import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"HandLandmarker 模型不存在: {MODEL_PATH}，请下载 "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style="white"):
        h, w = image.shape[:2]
        pts = [(int(round(lm.x * w)), int(round(lm.y * h))) for lm in keypoint_2d]
        if style == "default":
            connection_color = (0, 255, 0)
            connection_thickness = 2
            landmark_color = (0, 0, 255)
            landmark_radius = 3
        else:
            connection_color = (224, 224, 224)
            connection_thickness = 2
            landmark_color = (255, 48, 48)
            landmark_radius = 4
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                image, pts[start], pts[end], connection_color, connection_thickness
            )
        for pt in pts:
            cv2.circle(image, pt, landmark_radius, landmark_color, -1)
        return image

    def detect(self, rgb):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        results = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        if not results.hand_landmarks:
            return 0, None, None, None

        desired_hand_num = -1
        for i in range(len(results.hand_landmarks)):
            label = results.handedness[i][0].category_name
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None

        keypoint_3d = results.hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.hand_landmarks[desired_hand_num]
        num_box = len(results.hand_landmarks)

        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d[i].x
            keypoint[i][1] = keypoint_3d[i].y
            keypoint[i][2] = keypoint_3d[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d[i].x
            keypoint[i][1] = keypoint_2d[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        x_vector = points[0] - points[2]

        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
