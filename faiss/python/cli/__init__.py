# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Faissctl CLI - Unified command-line interface for Faiss IPC management.

This package provides tools for managing GPU memory pools and IVF lists
via shared memory IPC, similar to kvctl for kvcached.
"""

__version__ = "1.0.0"

from .utils import format_size, parse_size, supports_color, clr

__all__ = [
    "format_size",
    "parse_size",
    "supports_color",
    "clr",
]
