import sys
import threading
import itertools
import time

frames = [
    "▐|\\________▌",
    "▐__|\\______▌",
    "▐____|\\____▌",
    "▐______|\\__▌",
    "▐________|\\▌",
    "▐_______/|_▌",
    "▐_____/|___▌",
    "▐___/|_____▌",
    "▐/|________▌",
]


class Spinner:
    def __init__(self, text="", spin_chars=frames):
        self.text = text
        self.spin_chars = spin_chars
        self.spinning = False
        self.spinner_thread = None
        self._stream = sys.stdout
        self.is_tty = self._stream.isatty()
        self.max_text_length = len(text)

    def _spin(self):
        for char in itertools.cycle(self.spin_chars):
            if not self.spinning:
                break
            if self.is_tty:
                # Clear the line first
                max_spinner_width = max(len(frame) for frame in self.spin_chars)
                self._stream.write(
                    f"\r{' ' * (max_spinner_width + self.max_text_length + 1)}\r"
                )
                # Write the new content
                self._stream.write(f"{char} {self.text}")
                self._stream.flush()
            time.sleep(0.213)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        if not self.is_tty:
            print(f"* {self.text}")
            return
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.start()

    def update_text(self, new_text: str):
        """Update the spinner text while it's running"""
        self.text = new_text
        # Track the maximum text length for proper cleanup
        self.max_text_length = max(self.max_text_length, len(new_text))

    def pause(self):
        """Temporarily pause the spinner and clear the line"""
        if not self.spinning:
            return

        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()

        if self.is_tty:
            # Clear the current spinner line
            self._stream.write("\r")
            max_spinner_width = max(len(frame) for frame in self.spin_chars)
            self._stream.write(" " * (max_spinner_width + self.max_text_length + 1))
            self._stream.write("\r")
            self._stream.flush()

    def resume(self):
        """Resume the spinner after it was paused"""
        if self.spinning or not self.is_tty:
            return

        self.spinning = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.start()

    def stop(self):
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        if self.is_tty:
            self._stream.write("\r")
            # Account for maximum spinner width + maximum text length + spacing
            max_spinner_width = max(len(frame) for frame in self.spin_chars)
            self._stream.write(" " * (max_spinner_width + self.max_text_length + 1))
            self._stream.write("\r")
            self._stream.flush()
