"""Pytest configuration and environment fixtures for InspectAI.

Provides Windows-safe terminal handle resilience for Colorama when imported
via PyTorch/tqdm in headless CI/sandbox test runners.
"""

import os
import sys

# 1. Patch colorama.win32._winapi_test immediately to safely handle invalid Windows handles (0xc0000008)
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

# 2. Prevent OpenMP runtime conflict on Windows between PyTorch and scikit-learn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 3. Pre-load scientific stack in exact order (numpy -> scipy -> sklearn -> torch)
# This prevents Windows Error 1114 when sklearn dynamically loads vcomp140.dll
try:
    import numpy
    import scipy
    import sklearn
    import torch
except (ImportError, OSError):
    pass
