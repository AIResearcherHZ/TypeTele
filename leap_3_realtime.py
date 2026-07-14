from asr.tencent_asr import AsrServer
from hand_detect.detectFinger import FingerDetector
from retrieve.retrieve import Retrieve
from leap_hand_utils.leap_node import create_leap_node

import os
import time
import numpy as np
import cv2

from asr.typing_asr import KeyboardAsrServer
from ui import console, kv_panel

_REORDER_INDEX = np.array([9, 8, 10, 11, 5, 4, 6, 7, 1, 0, 2, 3, 12, 13, 14, 15])
_INVERSE_INDEX = np.argsort(_REORDER_INDEX)


class RealTimeRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.category = cfg["type"]["category"]
        self.curr_type = cfg["type"]["type_name"]
        self.open_pos, self.close_pos = self.load_type(self.curr_type)

        self.asr_type = cfg["asr"].get("type", "typing")
        if self.asr_type == "tencent":
            console.print("[cyan]提示:[/] 已启用腾讯云语音识别")
            self.asr = AsrServer(cfg["asr"])
        else:
            console.print("[cyan]提示:[/] 未启用语音识别，改用键盘输入")
            self.asr = KeyboardAsrServer(cfg["asr"])

        self.finger_detector = FingerDetector(cfg["detector"])
        self.retriever = Retrieve(
            api_key=cfg["retriever"]["api_key"],
            base_url=cfg["retriever"]["base_url"],
            category=self.category,
        )
        self.leap_node = create_leap_node(self.cfg["leap_cfg"])

        console.print(
            kv_panel(
                "LEAP 实时遥操作",
                [
                    (
                        "灵巧手后端",
                        "[bold]仿真 (MuJoCo)[/]"
                        if cfg["leap_cfg"].get("sim")
                        else "[bold]真实硬件[/]",
                        "",
                    ),
                    ("初始手势", self.curr_type, f"类别 {self.category}"),
                    (
                        "指令输入",
                        "腾讯云语音" if self.asr_type == "tencent" else "键盘",
                        "/手势名 可直接切换",
                    ),
                    (
                        "摄像头",
                        str(cfg["detector"]["camera"].get("camera_id", 0)),
                        f"检测 {cfg['detector'].get('hand_type', 'Right')} 手",
                    ),
                ],
                footer="画面窗口按 q 退出;Ctrl+C 停止",
            )
        )

    def start(self):
        self.asr.start()
        self.finger_detector.start()
        self.retriever.load_type_library()
        self.retriever.start()
        self.main_loop()

    def stop(self):
        self.asr.stop()
        self.finger_detector.stop()
        self.retriever.stop()
        self.leap_node.close()
        cv2.destroyAllWindows()

    def change_type(self, new_type: str):
        console.print(f"[cyan]提示:[/] 切换抓取手势: {self.curr_type} -> {new_type}")
        self.curr_type = new_type
        self.open_pos, self.close_pos = self.load_type(self.curr_type)

    def load_type(self, type_name: str):
        def parse_line(line: str):
            line = line.strip().strip("[]")
            parts = [p for p in line.replace(",", " ").split() if p]
            vals = [float(p) for p in parts]
            if len(vals) != 16:
                raise ValueError(f"Invalid line length for {type_name}: {len(vals)}")
            return np.array(vals, dtype=float)

        def _decode_saved(vec: np.ndarray) -> np.ndarray:
            joints = np.zeros(16, dtype=float)
            joints[_INVERSE_INDEX] = vec
            return joints + 3.14159

        current_dir = os.path.dirname(os.path.abspath(__file__))
        type_file = os.path.join(
            current_dir, "TypeLibrary", self.category, f"{type_name}.txt"
        )

        if not os.path.exists(type_file):
            raise FileNotFoundError(f"Type file not found: {type_file}")

        with open(type_file, "r", encoding="utf-8") as f:
            open_abs = _decode_saved(parse_line(f.readline()))
            close_abs = _decode_saved(parse_line(f.readline()))

        return open_abs, close_abs

    def main_loop(self):
        try:
            while True:
                if self.asr and self.asr.has_new_result():
                    new_query = self.asr.get()
                    if new_query:
                        console.print(f"[bold cyan]指令:[/] 收到新输入: {new_query}")
                        if new_query.startswith("/"):
                            try:
                                self.change_type(new_query[1:])
                                continue
                            except Exception as e:
                                console.print(f"[red]错误: 切换手势失败: {e}[/]")
                        self.retriever.retrieve(new_query)

                if self.retriever.has_new_result():
                    result = self.retriever.get()
                    if result and result != self.curr_type:
                        self.change_type(result)

                result = self.finger_detector.get()
                if result:
                    ratio, bgr = result

                    thumb_mask = np.array([0] * 12 + [1] * 4)
                    index_mask = np.array([1] * 4 + [0] * 12)
                    middle_mask = np.array([0] * 4 + [1] * 4 + [0] * 8)
                    ring_mask = np.array([0] * 8 + [1] * 4 + [0] * 4)

                    type_pos = (
                        (
                            self.open_pos * (1 - ratio["thumb"])
                            + self.close_pos * ratio["thumb"]
                        )
                        * thumb_mask
                        + (
                            self.open_pos * (1 - ratio["index"])
                            + self.close_pos * ratio["index"]
                        )
                        * index_mask
                        + (
                            self.open_pos * (1 - ratio["middle"])
                            + self.close_pos * ratio["middle"]
                        )
                        * middle_mask
                        + (
                            self.open_pos * (1 - ratio["ring"])
                            + self.close_pos * ratio["ring"]
                        )
                        * ring_mask
                    )

                    self.leap_node.set_leap(type_pos)

                    if bgr is not None:
                        cv2.imshow("Hand Detection", bgr)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                time.sleep(0.01)

        except KeyboardInterrupt:
            console.print("[dim]正在停止...[/]")
            self.stop()
            console.print("[green]已关闭。[/]")


def run_leap():
    cfg = {
        "asr": {
            "type": "typing",
            "verbose": True,
            "credentials": {
                "secret_id": "your_secret_id",
                "secret_key": "your_secret_key",
            },
            "audio": {"channels": 1, "sample_rate": 16000, "chunk_duration": 0.1},
            "vad": {
                "silence_threshold": 500,
                "min_audio_length": 0.5,
                "max_silence_duration": 2.0,
            },
            "tencent": {
                "endpoint": "asr.tencentcloudapi.com",
                "region": "ap-guangzhou",
                "engine_service_type": "16k_zh",
            },
            "test_microphone": True,
        },
        "retriever": {
            "api_key": "your_api_key",
            "base_url": "https://api.deepseek.com",
            "category": "leap",
        },
        "detector": {
            "camera": {
                "camera_id": 4,
                "width": 640,
                "height": 480,
                "fps": 30,
                "queue_size": 1,
            },
            "hand_type": "Left",
            "selfie": False,
        },
        "type": {"type_name": "box", "category": "leap"},
        "leap_cfg": {"sim": True, "curr_lim": 150, "kP": 100, "kI": 0, "kD": 150},
    }
    runner = RealTimeRunner(cfg)
    runner.start()


if __name__ == "__main__":
    run_leap()
