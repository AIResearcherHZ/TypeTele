import os

import cv2
import numpy as np

from hand_detect.detectFinger import FingerDetector
from hand2_utils.hand_node import NUM_JOINTS, create_hand_node
from ui import KeyReader, command_panel, console, pose_table

HAND_CATEGORY = "hand2"
CLOSE_SCALE = 0.8

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

CV_KEY_MAP = {
    ord(" "): " ",
    13: "\r",
    ord("r"): "r",
    ord("h"): "h",
    ord("q"): "q",
    27: "\x1b",
}


class Hand2CreateType:
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = "approach"
        self.approach_eef = None
        self.open_pos = None
        self.close_pos = None

        console.print("[dim]正在初始化 hand2 灵巧手...[/]")
        self.hand_node = create_hand_node(cfg["hand_cfg"])
        self.close_ref = self.build_close_ref()

        self.finger_detector = FingerDetector(cfg["detector"])
        self.finger_detector.start()

        self.enter_approach()

    def build_close_ref(self):
        import mujoco

        model = self.hand_node.model
        joint_ids = model.actuator_trnid[:NUM_JOINTS, 0]
        close_ref = np.zeros(NUM_JOINTS)
        for i, jid in enumerate(joint_ids):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if "abd" in name:
                continue
            close_ref[i] = CLOSE_SCALE * model.jnt_range[jid, 1]
        return close_ref

    def enter_approach(self):
        self.phase = "approach"
        self.approach_eef = None
        self.open_pos = None
        self.close_pos = None
        self.hand_node.enable_free_drag_mode()
        console.print(
            command_panel(
                "阶段 1/2: EEF 靠近 (手动拖拽)",
                [
                    ("拖拽", "在 MuJoCo 窗口双击选中基座/指节，Ctrl+右键拖拽靠近物体"),
                    ("空格", "记录靠近位姿 EEF，进入 pre-grasp 阶段"),
                    ("h", "显示帮助"),
                    ("q / Esc", "退出"),
                ],
                footer="热键在终端、MuJoCo 窗口和摄像头窗口都有效",
            )
        )

    def enter_pregrasp(self):
        self.approach_eef = self.hand_node.read_eef()
        self.hand_node.disable_free_drag_mode()
        self.hand_node.set_eef(self.approach_eef)
        self.phase = "pregrasp"
        eef_str = " ".join(f"{v:+.3f}" for v in self.approach_eef)
        console.print(f"[green]已记录靠近 EEF:[/] {eef_str}")
        console.print(
            command_panel(
                "阶段 2/2: pre-grasp (摄像头图像映射)",
                [
                    ("摄像头", "在摄像头前做手势，手指弯曲实时映射到仿真手"),
                    ("空格", "记录当前姿态 (第 1 次=张开，第 2 次=闭合)"),
                    ("回车", "保存手势"),
                    ("r", "重录 (回到 EEF 靠近阶段)"),
                    ("q / Esc", "退出"),
                ],
                footer="EEF 已固定，摄像头只控制手指",
            )
        )

    def apply_camera_mapping(self, ratio):
        r = np.repeat([ratio[f] for f in FINGERS], 4)
        self.hand_node.set_pos(r * self.close_ref)

    def record_next(self):
        if self.phase == "approach":
            self.enter_pregrasp()
            return
        pos = self.hand_node.read_pos()
        if self.open_pos is None:
            self.open_pos = pos
            console.print(pose_table("1/2 已记录「张开」姿态 (rad)", pos))
            console.print("[bold]在摄像头前摆好「闭合」手势后再按一次空格[/]")
        elif self.close_pos is None:
            self.close_pos = pos
            console.print(
                pose_table("2/2 已记录「闭合」姿态 (rad)", pos, style="magenta")
            )
            console.print("[bold]按回车保存，或按 r 重录[/]")
        else:
            console.print("[yellow]两个姿态都已录好，按回车保存，或按 r 重录[/]")

    def save_data(self, keys):
        if self.open_pos is None or self.close_pos is None:
            console.print("[yellow]还没录满两个姿态，先按空格记录[/]")
            return

        keys.pause()
        try:
            type_name = input("给这个手势起个名字: ").strip()
        finally:
            keys.resume()
        if not type_name:
            console.print("[yellow]名字为空，未保存[/]")
            return

        save_dir = os.path.join("TypeLibrary", HAND_CATEGORY)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{type_name}.txt")

        if os.path.exists(save_path):
            keys.pause()
            try:
                confirm = (
                    input(f"手势「{type_name}」已存在，是否覆盖？(y/n): ")
                    .strip()
                    .lower()
                )
            finally:
                keys.resume()
            if confirm != "y":
                console.print("[yellow]已取消保存[/]")
                return

        with open(save_path, "w") as f:
            f.write(" ".join(map(str, self.open_pos)) + "\n")
            f.write(" ".join(map(str, self.close_pos)) + "\n")
            f.write(" ".join(map(str, self.approach_eef)) + "\n")
            f.write(" ".join(map(str, self.approach_eef)) + "\n")

        console.print(f"[green]手势已保存到: {save_path}[/] (含靠近 EEF 位姿)")
        console.print("[bold]继续录下一个手势：回到 EEF 靠近阶段[/]")
        self.enter_approach()

    def print_help(self):
        if self.phase == "approach":
            console.print("[dim]阶段 1/2: 拖拽靠近物体，空格记录 EEF[/]")
        else:
            console.print("[dim]阶段 2/2: 摄像头控制手指，空格录姿态，回车保存[/]")

    def run(self):
        console.print()
        try:
            with KeyReader() as keys:
                while True:
                    result = self.finger_detector.get(timeout=0.02)
                    key = keys.poll()

                    if result:
                        ratio, bgr = result
                        if ratio is not None and self.phase == "pregrasp":
                            self.apply_camera_mapping(ratio)
                        if bgr is not None:
                            cv2.imshow("Hand Detection", bgr)
                            cv_key = cv2.waitKey(1) & 0xFF
                            if key is None and cv_key in CV_KEY_MAP:
                                key = CV_KEY_MAP[cv_key]

                    if key is None:
                        continue
                    if key == " ":
                        self.record_next()
                    elif key in ("\r", "\n", "s"):
                        self.save_data(keys)
                    elif key == "r":
                        console.print("[yellow]已清空，回到 EEF 靠近阶段[/]")
                        self.enter_approach()
                    elif key == "h":
                        self.print_help()
                    elif key in ("q", "\x1b"):
                        console.print("[dim]正在退出...[/]")
                        break
        except KeyboardInterrupt:
            console.print("\n[yellow]检测到 Ctrl+C，正在退出...[/]")

        self.cleanup()

    def cleanup(self):
        for step in (
            self.finger_detector.stop,
            cv2.destroyAllWindows,
            self._close_hand_node,
        ):
            try:
                step()
            except Exception as e:
                console.print(f"[red]清理资源失败: {e}[/]")

    def _close_hand_node(self):
        if self.hand_node is None:
            return
        if self.hand_node.free_drag_active:
            self.hand_node.disable_free_drag_mode()
        self.hand_node.close()


def main():
    cfg = {
        "hand_cfg": {"sim": True},
        "detector": {
            "camera": {
                "camera_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "queue_size": 1,
            },
            "hand_type": "Left",
            "selfie": False,
        },
    }

    recorder = Hand2CreateType(cfg)
    recorder.run()


if __name__ == "__main__":
    main()
