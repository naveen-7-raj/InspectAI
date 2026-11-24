"""Pytest configuration and environment fixtures for InspectAI.

Provides Windows-safe terminal handle resilience for Colorama when imported
via PyTorch/tqdm in headless CI/sandbox test runners.
"""

import sys

# Patch colorama.win32._winapi_test to safely handle invalid Windows handles (0xc0000008)
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


