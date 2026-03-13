# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for faissctl CLI.

Includes:
- Size parsing (human-readable to bytes)
- Size formatting (bytes to human-readable)
- ANSI color support
- Error handling utilities
"""

import os
import sys
from typing import Optional

# ANSI color codes
_ANSI_COLOR_CODES = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
}


def supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    if os.getenv('NO_COLOR') is not None:
        return False
    return sys.stdout.isatty()


def clr(text: str, color: Optional[str] = None, *, bold: bool = False) -> str:
    """
    Apply ANSI color to text.
    
    Args:
        text: Text to colorize
        color: Color name (red, green, yellow, blue, magenta, cyan)
        bold: Whether to apply bold formatting
        
    Returns:
        Colorized text (or original if colors not supported)
    """
    if not supports_color():
        return text
    
    seq = ''
    if bold:
        seq += _ANSI_COLOR_CODES['bold']
    if color and color in _ANSI_COLOR_CODES:
        seq += _ANSI_COLOR_CODES[color]
    
    if not seq:
        return text
    
    return f"{seq}{text}{_ANSI_COLOR_CODES['reset']}"


# Size suffixes for parsing/formatting
SIZE_SUFFIXES = {
    'b': 1,
    'k': 1024,
    'kb': 1024,
    'm': 1024 ** 2,
    'mb': 1024 ** 2,
    'g': 1024 ** 3,
    'gb': 1024 ** 3,
    't': 1024 ** 4,
    'tb': 1024 ** 4,
}


def parse_size(size_str: str) -> int:
    """
    Parse human-readable size string to bytes.
    
    Supports: B, K/KB, M/MB, G/GB, T/TB suffixes (case-insensitive)
    Also supports plain numbers (interpreted as bytes)
    
    Args:
        size_str: Size string like "512M", "2G", "1024"
        
    Returns:
        Size in bytes
        
    Raises:
        ValueError: If size string is invalid
    """
    if not size_str:
        raise ValueError("Size string cannot be empty")
    
    s = size_str.strip().lower().replace(',', '').replace('_', '')
    
    # Try to match longest suffix first
    for suf, mul in sorted(SIZE_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suf):
            num_part = s[:-len(suf)] or "0"
            try:
                num = float(num_part)
            except ValueError as exc:
                raise ValueError(f"Invalid size string '{size_str}'") from exc
            return int(num * mul)
    
    # No suffix - assume raw bytes
    try:
        return int(float(s))
    except ValueError as exc:
        raise ValueError(f"Invalid size string '{size_str}'") from exc


def format_size(num_bytes: int, precision: int = 2) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        num_bytes: Size in bytes
        precision: Number of decimal places
        
    Returns:
        Formatted string like "1.50 GB"
    """
    if num_bytes < 0:
        return f"-{format_size(-num_bytes, precision)}"
    
    if num_bytes == 0:
        return "0 B"
    
    size: float = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.{precision}f} {unit}"
        size /= 1024
    
    return f"{size:.{precision}f} {units[-1]}"


def format_percentage(used: float, total: float) -> str:
    """
    Format percentage with color coding.
    
    Args:
        used: Used amount
        total: Total amount
        
    Returns:
        Colorized percentage string
    """
    if total <= 0:
        return clr("N/A", 'yellow')
    
    pct = (used / total) * 100
    
    if pct < 50:
        color = 'green'
    elif pct < 80:
        color = 'yellow'
    else:
        color = 'red'
    
    return clr(f"{pct:.1f}%", color)


class CliError(Exception):
    """Base exception for CLI errors."""
    pass


class PoolNotFoundError(CliError):
    """Raised when GPU pool is not found."""
    pass


class IndexNotFoundError(CliError):
    """Raised when index file is not found."""
    pass


class IpcError(CliError):
    """Raised when IPC communication fails."""
    pass


def print_error(message: str) -> None:
    """Print error message to stderr with red color."""
    print(clr(f"Error: {message}", 'red', bold=True), file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message with yellow color."""
    print(clr(f"Warning: {message}", 'yellow'))


def print_success(message: str) -> None:
    """Print success message with green color."""
    print(clr(message, 'green'))
