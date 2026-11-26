"""Root pytest configuration and environment setup for InspectAI.

Pre-configures environment variables and loads Windows OpenMP runtimes in the
exact required sequence before test discovery starts, preventing WinError 1114.
"""

import os
import sys

# 1. Prevent OpenMP runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Patch colorama Windows handle resilience for headless runners
try:
    import colorama.win32
    orig_winapi_test = colorama.win32._winapi_test

    def safe_winapi_test(handle):
        try:
            return orig_winapi_test(handle)
        except OSError:
            return False

    colorama.win32._winapi_test = safe_winapi_test
except (ImportError, AttributeError):
    pass

# 3. Pre-load scientific stack before any test file imports
try:
    import numpy
    import scipy
    import sklearn
    import torch
except (ImportError, OSError):
    pass
