import threading

import cv2
import numpy as np

from ui import console

from .Camera import Camera
from .SingleHandDetetor import SingleHandDetector

FINGER_RANGES = (
    ("thumb", 4, 0.02, 0.09),
    ("index", 8, 0.09, 0.17),
    ("middle", 12, 0.09, 0.18),
    ("ring", 16, 0.07, 0.17),
    ("pinky", 20, 0.08, 0.14),
)


class FingerDetector:
    def __init__(self, cfg):
        self.cam = Camera(cfg["camera"])
        self.detector = SingleHandDetector(
            hand_type=cfg.get("hand_type", "Right"), selfie=cfg.get("selfie", True)
        )

        self._latest = None
        self._cond = threading.Condition()
        self.running = False
        self.detection_thread = None

    def _detection_loop(self):
        while self.running:
            bgr = self.cam.get_frame()
            if bgr is None:
                continue
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                _, joint_pos, keypoint_2d, _ = self.detector.detect(rgb=rgb)
            except Exception as e:
                console.print(f"[red]手部检测出错: {e}[/]")
                continue

            finger_ratios = None
            if joint_pos is not None:
                bgr = self.detector.draw_skeleton_on_image(
                    bgr, keypoint_2d, style="default"
                )
                finger_ratios = {}
                for name, tip, lo, hi in FINGER_RANGES:
                    vec = joint_pos[tip] - joint_pos[0]
                    if name == "thumb":
                        vec[0] = vec[2] = 0
                    length = np.linalg.norm(vec)
                    finger_ratios[name] = 1 - np.clip((length - lo) / (hi - lo), 0, 1)

            with self._cond:
                self._latest = (finger_ratios, bgr)
                self._cond.notify_all()

    def start(self):
        if self.running:
            return
        self.cam.start()
        self.running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.detection_thread.start()
        console.print("[green]实时手部检测已启动！[/]")

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.cam.stop()
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3.0)
        self.detector.close()
        console.print("[dim]实时手部检测已停止！[/]")

    def get(self, timeout=0.0):
        with self._cond:
            if self._latest is None and timeout > 0:
                self._cond.wait(timeout)
            result, self._latest = self._latest, None
        return result

    def is_running(self):
        return self.running
