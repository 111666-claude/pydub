import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(ROOT, "pydub")

# pydub.utils falls back to a bare `import pyaudioop`, which requires the
# package directory itself to be importable.
for path in (ROOT, PACKAGE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
