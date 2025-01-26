import sys
import threading
import itertools
import time


class Spinner:
    def __init__(self, text="", spin_chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        self.text = text
        self.spin_chars = spin_chars
        self.spinning = False
        self.spinner_thread = None
        self._stream = sys.stdout
        self.is_tty = self._stream.isatty()

    def _spin(self):
        for char in itertools.cycle(self.spin_chars):
            if not self.spinning:
                break
            if self.is_tty:
                self._stream.write(f"\r{char} {self.text}")
                self._stream.flush()
            time.sleep(0.1)

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

    def stop(self):
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        if self.is_tty:
            self._stream.write("\r")
            self._stream.write(" " * (len(self.text) + 2))
            self._stream.write("\r")
            self._stream.flush()
