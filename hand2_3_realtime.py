from asr.tencent_asr import AsrServer
from hand_detect.detectFinger import FingerDetector
from retrieve.retrieve import Retrieve
from hand2_utils.hand_node import NUM_JOINTS, create_hand_node

import os
import numpy as np
import cv2

from asr.typing_asr import KeyboardAsrServer
from ui import console, kv_panel

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class RealTimeRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.category = cfg["type"]["category"]
        self.curr_type = cfg["type"]["type_name"]
        self.open_pos, self.close_pos, self.open_eef, self.close_eef = self.load_type(
            self.curr_type
        )

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
        self.hand_node = create_hand_node(self.cfg["hand_cfg"])
        self.hand_node.set_eef(self.open_eef)

        console.print(
            kv_panel(
                "hand2 实时遥操作",
                [
                    (
                        "灵巧手后端",
                        "[bold]仿真 (MuJoCo)[/]",
                        "assets/hand2/scene_right.xml",
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
        self.hand_node.close()
        cv2.destroyAllWindows()

    def change_type(self, new_type: str):
        console.print(f"[cyan]提示:[/] 切换抓取手势: {self.curr_type} -> {new_type}")
        self.curr_type = new_type
        self.open_pos, self.close_pos, self.open_eef, self.close_eef = self.load_type(
            self.curr_type
        )
        self.hand_node.set_eef(self.open_eef)

    def get_eef(self):
        return self.hand_node.read_eef()

    def load_type(self, type_name: str):
        def parse_line(line: str, expected: int):
            line = line.strip().strip("[]")
            parts = [p for p in line.replace(",", " ").split() if p]
            vals = [float(p) for p in parts]
            if len(vals) != expected:
                raise ValueError(f"Invalid line length for {type_name}: {len(vals)}")
            return np.array(vals, dtype=float)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        type_file = os.path.join(
            current_dir, "TypeLibrary", self.category, f"{type_name}.txt"
        )

        if not os.path.exists(type_file):
            raise FileNotFoundError(f"Type file not found: {type_file}")

        with open(type_file, "r", encoding="utf-8") as f:
            open_pos = parse_line(f.readline(), NUM_JOINTS)
            close_pos = parse_line(f.readline(), NUM_JOINTS)
            open_eef = parse_line(f.readline(), 7)
            close_eef = parse_line(f.readline(), 7)

        return open_pos, close_pos, open_eef, close_eef

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

                result = self.finger_detector.get(timeout=0.02)
                if result:
                    ratio, bgr = result

                    if ratio is not None:
                        r = np.repeat([ratio[f] for f in FINGERS], 4)
                        self.hand_node.set_pos(
                            self.open_pos * (1 - r) + self.close_pos * r
                        )

                    if bgr is not None:
                        cv2.imshow("Hand Detection", bgr)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

        except KeyboardInterrupt:
            console.print("[dim]正在停止...[/]")
            self.stop()
            console.print("[green]已关闭。[/]")


def run_hand2():
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
            "category": "hand2",
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
        "type": {"type_name": "grasp", "category": "hand2"},
        "hand_cfg": {"sim": True},
    }
    runner = RealTimeRunner(cfg)
    runner.start()


if __name__ == "__main__":
    run_hand2()
