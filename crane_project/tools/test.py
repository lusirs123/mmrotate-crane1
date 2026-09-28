"""Compatibility entry point for the repository legacy MMRotate test tool.

The server uses MMCV/MMDetection 1.x.  The previous implementation imported
MMEngine-only APIs and failed before loading a config.
"""

from pathlib import Path
import runpy


if __name__ == '__main__':
    legacy = Path(__file__).resolve().parents[2] / 'tools' / 'test.py'
    runpy.run_path(str(legacy), run_name='__main__')
