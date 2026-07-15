import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hand2_utils.hand_node import NUM_JOINTS, create_hand_node
from ui import KeyReader, command_panel, console, pose_table

HAND_CATEGORY = "hand2"


def load_type(type_name: str, category: str = HAND_CATEGORY):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "TypeLibrary")
    type_library_path = os.path.join(base_path, category)

    if not os.path.isdir(type_library_path):
        raise FileNotFoundError(f"Type directory not found: {type_library_path}")

    type_file = os.path.join(type_library_path, f"{type_name}.txt")
    if not os.path.exists(type_file):
        raise FileNotFoundError(f"Gesture file not found: {type_file}")

    def parse_line(line: str, expected: int):
        line = line.strip().strip("[]")
        parts = [p for p in line.replace(",", " ").split() if p]
        vals = [float(p) for p in parts]
        if len(vals) != expected:
            raise ValueError(
                f"Invalid line length (expected {expected}): {line} -> {len(vals)}"
            )
        return np.array(vals, dtype=float)

    with open(type_file, "r") as f:
        open_vec = parse_line(f.readline(), NUM_JOINTS)
        close_vec = parse_line(f.readline(), NUM_JOINTS)
        open_eef = parse_line(f.readline(), 7)
        close_eef = parse_line(f.readline(), 7)

    return open_vec, close_vec, open_eef, close_eef


class Hand2TypePlayer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.type_name = cfg["type"]["type_name"]
        self.hand_node = None
        self.open_saved = None
        self.close_saved = None
        self.open_eef = None
        self.close_eef = None
        self._eef_slerp = None
        self.fraction = 0.0
        self.step_size = 0.05

        self._init_hand()
        self._load_type()
        self._apply_fraction(0.0)

    def _init_hand(self):
        try:
            console.print("[dim]正在初始化 hand2 灵巧手...[/]")
            self.hand_node = create_hand_node(self.cfg["hand_cfg"])
            console.print("[green]hand2 灵巧手初始化成功[/]")
        except Exception as e:
            console.print(f"[red]灵巧手初始化失败: {e}[/]")
            raise

    def _load_type(self):
        try:
            console.print(f"[dim]正在加载手势: {self.type_name}[/]")
            o, c, oe, ce = load_type(self.type_name, HAND_CATEGORY)
            self.open_saved = o
            self.close_saved = c
            self.open_eef = oe
            self.close_eef = ce
            self._eef_slerp = Slerp(
                [0.0, 1.0],
                Rotation.from_quat([oe[[4, 5, 6, 3]], ce[[4, 5, 6, 3]]]),
            )
            console.print("[green]手势加载成功 (含 EEF 基座位姿)[/]")
            console.print(pose_table("「张开」姿态 (rad)", o))
            console.print(pose_table("「闭合」姿态 (rad)", c, style="magenta"))
        except Exception as e:
            console.print(f"[red]加载手势「{self.type_name}」失败: {e}[/]")
            raise

    def _apply_fraction(self, frac: float):
        if (
            self.hand_node is None
            or self.open_saved is None
            or self.close_saved is None
        ):
            return

        frac = max(0.0, min(1.0, frac))
        self.fraction = frac

        target_pos = self.open_saved * (1 - frac) + self.close_saved * frac
        self.hand_node.set_pos(target_pos)

        eef_xyz = self.open_eef[:3] * (1 - frac) + self.close_eef[:3] * frac
        quat_xyzw = self._eef_slerp(frac).as_quat()
        self.hand_node.set_eef(np.concatenate([eef_xyz, quat_xyzw[[3, 0, 1, 2]]]))

    def decrease(self):
        new_frac = self.fraction - self.step_size
        new_frac = max(0.0, new_frac)
        self._apply_fraction(new_frac)
        self._print_status()

    def increase(self):
        new_frac = self.fraction + self.step_size
        new_frac = min(1.0, new_frac)
        self._apply_fraction(new_frac)
        self._print_status()

    def set_fraction(self, frac: float):
        self._apply_fraction(frac)
        self._print_status()

    def _print_status(self):
        bar_length = 40
        filled = int(bar_length * self.fraction)
        bar = "█" * filled + "░" * (bar_length - filled)
        console.print(
            f"[{bar}] {self.fraction:.2f} (0=张开, 1=闭合)", end="\r", highlight=False
        )

    def print_help(self):
        console.print(
            command_panel(
                "hand2 手势回放工具",
                [
                    ("a / d", "张开一点 / 闭合一点 (可连按，每次 5%)"),
                    ("0 / 1", "一键全张开 / 全闭合"),
                    ("h", "显示本帮助"),
                    ("q / Esc", "退出"),
                ],
                footer=f"当前手势: {self.type_name}；热键在终端和 MuJoCo 窗口都有效",
            )
        )

    def run(self):
        console.print()
        self.print_help()
        self._print_status()

        try:
            with KeyReader() as keys:
                while True:
                    key = keys.poll(timeout=0.2)
                    if key is None:
                        continue
                    if key == "a":
                        self.decrease()
                    elif key == "d":
                        self.increase()
                    elif key == "0":
                        self.set_fraction(0.0)
                    elif key == "1":
                        self.set_fraction(1.0)
                    elif key == "h":
                        console.print()
                        self.print_help()
                    elif key in ("q", "\x1b"):
                        console.print("\n[dim]正在退出...[/]")
                        break
        except KeyboardInterrupt:
            console.print("\n[yellow]检测到 Ctrl+C，正在退出...[/]")

        self.cleanup()

    def cleanup(self):
        try:
            if self.hand_node and self.hand_node.free_drag_active:
                console.print("[dim]正在关闭自由拖拽模式...[/]")
                self.hand_node.disable_free_drag_mode()
                console.print("[green]自由拖拽模式已关闭[/]")
            if self.hand_node:
                self.hand_node.close()
        except Exception as e:
            console.print(f"[red]清理灵巧手资源失败: {e}[/]")


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "grasp"

    cfg = {
        "hand_cfg": {"sim": True},
        "type": {"type_name": name, "category": HAND_CATEGORY},
    }

    try:
        player = Hand2TypePlayer(cfg)
        player.run()
    except Exception as e:
        console.print(f"[bold red]程序异常退出: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
