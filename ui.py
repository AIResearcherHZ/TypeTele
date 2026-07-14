import select
import sys
import termios
import tty
from collections import deque

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

VIEWER_KEYS = deque()

GLFW_KEY_ESCAPE = 256
GLFW_KEY_ENTER = 257
GLFW_KEY_KP_ENTER = 335


def viewer_key_callback(keycode):
    if keycode == GLFW_KEY_ESCAPE:
        VIEWER_KEYS.append("\x1b")
    elif keycode in (GLFW_KEY_ENTER, GLFW_KEY_KP_ENTER):
        VIEWER_KEYS.append("\n")
    elif 32 <= keycode <= 126:
        VIEWER_KEYS.append(chr(keycode).lower())


class KeyReader:
    def __init__(self):
        self.enabled = bool(sys.stdin and sys.stdin.isatty())
        self._old = None

    def __enter__(self):
        if self.enabled:
            self._old = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *_):
        self.pause()

    def pause(self):
        if self.enabled and self._old is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            except termios.error:
                pass

    def resume(self):
        if self.enabled:
            tty.setcbreak(sys.stdin.fileno())

    def poll(self):
        if VIEWER_KEYS:
            return VIEWER_KEYS.popleft()
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        return ch.lower() if ch else None


FINGER_NAMES = ("无名指", "中指", "食指", "拇指")
JOINT_COLS = ("关节1", "关节2", "关节3", "关节4")


def command_panel(title, rows, footer=None):
    t = Table(box=box.ROUNDED, show_header=False, pad_edge=False, expand=False)
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="dim")
    for cmd, desc in rows:
        t.add_row(cmd, desc)
    head = Text(title, style="bold white")
    foot = Text(footer, style="dim italic") if footer else None
    return Panel(t, title=head, subtitle=foot, box=box.HEAVY, expand=False)


def pose_table(title, saved_pos, style="green"):
    t = Table(
        title=f"[bold {style}]{title}[/]",
        box=box.ROUNDED,
        pad_edge=False,
        expand=False,
        title_justify="left",
        header_style="dim",
    )
    t.add_column("手指", style="bold", no_wrap=True)
    for col in JOINT_COLS:
        t.add_column(col, justify="right")
    for i, name in enumerate(FINGER_NAMES):
        vals = saved_pos[i * 4 : (i + 1) * 4]
        t.add_row(name, *[f"{v:+.3f}" for v in vals])
    return t


def kv_panel(title, rows, footer=None):
    t = Table(box=box.ROUNDED, show_header=False, pad_edge=False, expand=False)
    t.add_column(style="bold", no_wrap=True)
    t.add_column(justify="right")
    t.add_column(style="dim")
    for row in rows:
        key, val = row[0], row[1]
        note = row[2] if len(row) > 2 else ""
        t.add_row(key, val, note)
    head = Text(title, style="bold white")
    foot = Text(footer, style="dim italic") if footer else None
    return Panel(t, title=head, subtitle=foot, box=box.HEAVY, expand=False)
