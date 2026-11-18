"""ANSI status lines in the style of the AMR study runner."""

from __future__ import annotations

import os
import sys

_COLOR = False


def enable_color() -> None:
    global _COLOR
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        _COLOR = False
        return
    if sys.platform == "win32":
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                _COLOR = False
                return
            kernel.SetConsoleMode(handle, mode.value | 0x0004)
        except (AttributeError, OSError):
            _COLOR = False
            return
    _COLOR = True


def paint(text: str, code: str) -> str:
    if not _COLOR or not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def title(text: str) -> None:
    print(flush=True)
    print(paint(text, "1;32"), flush=True)


def line(text: str) -> None:
    print(f"  {text}", flush=True)


class Status:
    """One updating line per stage, under a bold section title."""

    def __init__(self) -> None:
        self._section: str | None = None
        self._short: str | None = None
        self._width = 0

    def update(self, section: str, short: str, done: int, total: int) -> None:
        if short != self._short:
            self.close()
            if section != self._section:
                title(section)
                self._section = section
            self._short = short
        filled = 0 if total <= 0 else int(20 * done / total)
        bar = paint("#" * filled, "32") + paint("-" * (20 - filled), "2")
        visible = f"  {short:<18} [{'#' * filled}{'-' * (20 - filled)}] {done}/{total}"
        text = f"  {short:<18} [{bar}] {done}/{total}"
        padding = " " * max(0, self._width - len(visible))
        print("\r" + text + padding, end="", flush=True)
        self._width = len(visible)

    def close(self) -> None:
        if self._short is not None:
            print(flush=True)
            self._short = None
            self._width = 0
