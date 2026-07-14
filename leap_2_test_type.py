import os
import sys
import time

import numpy as np

from leap_hand_utils.leap_node import create_leap_node
from ui import KeyReader, command_panel, console, pose_table

HAND_CATEGORY = "leap"

_REORDER_INDEX = np.array([9, 8, 10, 11, 5, 4, 6, 7, 1, 0, 2, 3, 12, 13, 14, 15])
_INVERSE_INDEX = np.argsort(_REORDER_INDEX)


def _decode_saved(vec: np.ndarray) -> np.ndarray:
    joints = np.zeros(16, dtype=float)
    joints[_INVERSE_INDEX] = vec
    pos = joints + 3.14159
    return pos


def load_type(type_name: str, category: str = HAND_CATEGORY):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "TypeLibrary")
    type_library_path = os.path.join(base_path, category)

    if not os.path.isdir(type_library_path):
        raise FileNotFoundError(f"Type directory not found: {type_library_path}")

    type_file = os.path.join(type_library_path, f"{type_name}.txt")
    if not os.path.exists(type_file):
        raise FileNotFoundError(f"Gesture file not found: {type_file}")

    with open(type_file, "r") as f:
        open_line = f.readline().strip()
        close_line = f.readline().strip()

    def parse_line(line: str):
        line = line.strip().strip("[]")
        parts = [p for p in line.replace(",", " ").split() if p]
        vals = [float(p) for p in parts]
        if len(vals) != 16:
            raise ValueError(
                f"Invalid line length (expected 16): {line} -> {len(vals)}"
            )
        return np.array(vals, dtype=float)

    open_vec = parse_line(open_line)
    close_vec = parse_line(close_line)

    return open_vec, close_vec


class LeapTypePlayer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.type_name = cfg["type"]["type_name"]
        self.leap_node = None
        self.open_saved = None
        self.close_saved = None
        self.fraction = 0.0
        self.step_size = 0.05

        self._init_leap()
        self._load_type()
        self._apply_fraction(0.0)

    def _init_leap(self):
        try:
            console.print("[dim]正在初始化 LEAP 灵巧手...[/]")
            self.leap_node = create_leap_node(self.cfg["leap_cfg"])
            console.print("[green]LEAP 灵巧手初始化成功[/]")
        except Exception as e:
            console.print(f"[red]灵巧手初始化失败: {e}[/]")
            raise

    def _load_type(self):
        try:
            console.print(f"[dim]正在加载手势: {self.type_name}[/]")
            o, c = load_type(self.type_name, HAND_CATEGORY)
            self.open_saved = o
            self.close_saved = c
            console.print("[green]手势加载成功[/]")
            console.print(pose_table("「张开」姿态 (rad)", o))
            console.print(pose_table("「闭合」姿态 (rad)", c, style="magenta"))
        except Exception as e:
            console.print(f"[red]加载手势「{self.type_name}」失败: {e}[/]")
            raise

    def _apply_fraction(self, frac: float):
        if (
            self.leap_node is None
            or self.open_saved is None
            or self.close_saved is None
        ):
            return

        frac = max(0.0, min(1.0, frac))
        self.fraction = frac

        interp_saved = self.open_saved * (1 - frac) + self.close_saved * frac
        target_pos = _decode_saved(interp_saved)

        self.leap_node.set_leap(target_pos)

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
                "LEAP 手势回放工具",
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
                    key = keys.poll()
                    if key is None:
                        time.sleep(0.02)
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
            if self.leap_node and self.leap_node.free_drag_active:
                console.print("[dim]正在关闭自由拖拽模式...[/]")
                self.leap_node.disable_free_drag_mode()
                console.print("[green]自由拖拽模式已关闭[/]")
            if self.leap_node:
                self.leap_node.close()
        except Exception as e:
            console.print(f"[red]清理灵巧手资源失败: {e}[/]")


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "processed_tape"

    cfg = {
        "leap_cfg": {"sim": True, "curr_lim": 150, "kP": 250, "kI": 0, "kD": 100},
        "type": {"type_name": name, "category": HAND_CATEGORY},
    }

    try:
        player = LeapTypePlayer(cfg)
        player.run()
    except Exception as e:
        console.print(f"[bold red]程序异常退出: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
