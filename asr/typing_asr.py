import sys
import threading
from queue import Empty, Full, Queue

from pynput import keyboard

from ui import console


class KeyboardAsrServer:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.verbose = cfg.get("verbose", True) if cfg else True

        self.is_running = False
        self.input_thread = None

        self.result_queue = Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self.have_new_result = False

        self.input_buffer = ""
        self.pynput_char_queue = Queue(maxsize=10)
        self.keyboard_listener = None

        console.print("[bold]键盘输入模式[/]  [dim]输入指令回车发送；quit/exit 退出[/]")

    def on_release(self, key):
        if not self.is_running:
            return

        char = None
        try:
            char = key.char
        except AttributeError:
            if key == keyboard.Key.enter:
                char = "\n"
            elif key == keyboard.Key.backspace:
                char = "\x7f"
            elif key == keyboard.Key.esc:
                char = "\x03"

        if char is not None:
            try:
                self.pynput_char_queue.put_nowait(char)
            except Full:
                pass

    def _input_loop(self):
        console.print("[green]键盘输入就绪，可以开始输入指令了...[/]")

        self.keyboard_listener = keyboard.Listener(
            on_release=self.on_release, daemon=True
        )
        self.keyboard_listener.start()

        while self.is_running:
            try:
                char = self.pynput_char_queue.get(timeout=0.2)
            except Empty:
                continue

            if char == "\n":
                if self.input_buffer.strip():
                    text = self.input_buffer.strip()
                    self.input_buffer = ""

                    if text.lower() in ["quit", "exit"]:
                        console.print(f"\n[yellow]检测到退出指令: {text}[/]")
                        self.is_running = False
                        break

                    self._put_result(text)
                    console.print(f"\n[green]指令已发送: {text}[/]")
                    console.print("请输入下一条指令: ", end="")
                else:
                    console.print("\n请继续输入: ", end="")

            elif char == "\x7f":
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()

            elif char == "\x03":
                console.print("\n[yellow]收到 Ctrl+C，正在退出...[/]")
                self.is_running = False
                break

            else:
                self.input_buffer += char
                sys.stdout.write(char)
                sys.stdout.flush()

        if self.keyboard_listener and self.keyboard_listener.is_alive():
            self.keyboard_listener.stop()

    def _put_result(self, text):
        with self._result_lock:
            if not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except Empty:
                    pass

            self.result_queue.put(text)
            self.have_new_result = True

    def start(self):
        if self.is_running:
            if self.verbose:
                console.print("[yellow]键盘输入已经在运行了[/]")
            return True

        self.is_running = True

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        if self.verbose:
            console.print("[green]键盘输入已启动[/]")

        return True

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)

        if self.verbose:
            console.print("[dim]键盘输入已停止[/]")

    def get(self):
        try:
            with self._result_lock:
                if self.have_new_result:
                    result = self.result_queue.get_nowait()
                    self.have_new_result = False
                    return result
                return None
        except Empty:
            return None

    def has_new_result(self):
        with self._result_lock:
            return self.have_new_result

    def is_running_status(self):
        return self.is_running
