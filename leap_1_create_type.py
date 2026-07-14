import os
import time

import numpy as np

from leap_hand_utils.leap_node import create_leap_node
from ui import KeyReader, command_panel, console, pose_table

HAND_CATEGORY = "leap"


class LeapCreateType:
    def __init__(self, cfg):
        self.cfg = cfg
        self.leap_node = None
        self.open_pos = None
        self.close_pos = None
        self.init_leap()

    def init_leap(self):
        console.print("[dim]正在初始化 LEAP 灵巧手...[/]")
        self.leap_node = create_leap_node(self.cfg["leap_cfg"])
        self.leap_node.enable_free_drag_mode()
        console.print("[green]LEAP 灵巧手已进入自由拖拽模式[/]")

    def _get_current_leap_pos(self):
        current_pos = self.leap_node.read_pos()
        joints = current_pos - 3.14159
        reorder_index = np.array([9, 8, 10, 11, 5, 4, 6, 7, 1, 0, 2, 3, 12, 13, 14, 15])
        inverse_index = np.argsort(reorder_index)
        reordered = np.array(joints)[inverse_index]
        return reordered

    def record_next(self):
        pos = self._get_current_leap_pos()
        if self.open_pos is None:
            self.open_pos = pos
            console.print(pose_table("1/2 已记录「张开」姿态 (rad)", pos))
            console.print("[bold]摆好「闭合」姿势后再按一次空格[/]")
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

        console.print(f"[green]手势已保存到: {save_path}[/]")
        self.open_pos = None
        self.close_pos = None
        console.print("[bold]可以继续录下一个手势：摆好「张开」姿势按空格[/]")

    def reset_positions(self):
        self.open_pos = None
        self.close_pos = None
        console.print("[yellow]已清空，重新摆好「张开」姿势按空格[/]")

    def print_help(self):
        console.print(
            command_panel(
                "LEAP 手势录制工具",
                [
                    ("空格", "记录当前姿态 (第 1 次=张开，第 2 次=闭合)"),
                    ("回车", "保存手势 (只在起名字时需要打字)"),
                    ("r", "重录当前手势"),
                    ("h", "显示本帮助"),
                    ("q / Esc", "退出"),
                ],
                footer="在 MuJoCo 窗口拖拽摆姿势；热键在终端和 MuJoCo 窗口都有效",
            )
        )
        console.print("[bold]摆好「张开」姿势后按空格开始[/]")

    def run(self):
        console.print()
        self.print_help()
        try:
            with KeyReader() as keys:
                while True:
                    key = keys.poll()
                    if key is None:
                        time.sleep(0.02)
                        continue
                    if key == " ":
                        self.record_next()
                    elif key in ("\r", "\n", "s"):
                        self.save_data(keys)
                    elif key == "r":
                        self.reset_positions()
                    elif key == "h":
                        self.print_help()
                    elif key in ("q", "\x1b"):
                        console.print("[dim]正在退出...[/]")
                        break
        except KeyboardInterrupt:
            console.print("\n[yellow]检测到 Ctrl+C，正在退出...[/]")

        self.cleanup()

    def cleanup(self):
        try:
            if self.leap_node and self.leap_node.free_drag_active:
                self.leap_node.disable_free_drag_mode()
            if self.leap_node:
                self.leap_node.close()
        except Exception as e:
            console.print(f"[red]清理灵巧手资源失败: {e}[/]")


def main():
    cfg = {"leap_cfg": {"sim": True, "curr_lim": 120, "kP": 150, "kI": 0, "kD": 50}}

    recorder = LeapCreateType(cfg)
    recorder.run()


if __name__ == "__main__":
    main()
