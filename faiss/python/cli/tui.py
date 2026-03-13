# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TUI (Terminal User Interface) commands for faissctl.

Provides:
- watch: Non-interactive continuous monitoring
- top: Interactive curses-based TUI
"""

import subprocess
import sys
import time
from typing import List, Optional

from .pool_commands import detect_gpu_pools, get_pool_controller_safe
from .utils import clr, format_percentage, format_size, print_error


def _get_pool_stats(device_id: int) -> Optional[dict]:
    """Get pool stats, returning None on error."""
    try:
        ctrl = get_pool_controller_safe(device_id)
        return ctrl.get_stats()
    except Exception:
        return None


def _format_bar(used: float, total: float, width: int = 40) -> str:
    """Create ASCII bar for visualization."""
    if total <= 0:
        return " " * width
    
    ratio = used / total
    filled = int(ratio * width)
    
    if ratio < 0.5:
        color = 'green'
    elif ratio < 0.8:
        color = 'yellow'
    else:
        color = 'red'
    
    bar = "█" * filled + "░" * (width - filled)
    return clr(bar, color)


def cmd_watch(device_id: Optional[int], interval: float = 1.0, json_output: bool = False) -> int:
    """
    Watch pool statistics continuously (non-interactive).
    
    Similar to the 'watch' Unix command, this continuously refreshes
    pool statistics at a specified interval until interrupted.
    
    Args:
        device_id: Specific device to watch, or None for all
        interval: Update interval in seconds
        json_output: If True, output as JSON lines
        
    Returns:
        Exit code (0 for success, 1 for error, 130 for interrupt)
    """
    if json_output and device_id is None:
        print_error("JSON output requires a specific device_id")
        return 1
    
    try:
        while True:
            # Clear screen (Unix-like)
            if not json_output:
                sys.stdout.write("\033[2J\033[H")
            
            if device_id is not None:
                # Watch specific device
                stats = _get_pool_stats(device_id)
                if stats is None:
                    if not json_output:
                        print_error(f"Failed to get stats for GPU {device_id}")
                    else:
                        print(f'{{"error": "Failed to get stats for GPU {device_id}"}}')
                    return 1
                
                if json_output:
                    import json
                    print(json.dumps({
                        "device_id": device_id,
                        "timestamp": time.time(),
                        **stats,
                    }))
                    sys.stdout.flush()
                else:
                    # Print header
                    header = f"GPU {device_id} Pool - Every {interval:.1f}s (Press Ctrl+C to exit)"
                    print(clr(header, 'cyan', bold=True))
                    print("=" * 60)
                    print()
                    
                    # Print stats
                    total = stats["total_bytes"]
                    used = stats["used_bytes"]
                    available = stats["available_bytes"]
                    util = stats["utilization"]
                    
                    print(f"Total:     {format_size(total)}")
                    print(f"Used:      {format_size(used)} ({util * 100:.1f}%)")
                    print(f"Available: {format_size(available)}")
                    print()
                    
                    # Print bar
                    bar = _format_bar(used, total, width=50)
                    print(f"[{bar}] {format_percentage(used, total)}")
            else:
                # Watch all devices
                pools = detect_gpu_pools()
                
                header = f"All GPU Pools - Every {interval:.1f}s (Press Ctrl+C to exit)"
                print(clr(header, 'cyan', bold=True))
                print("=" * 70)
                print()
                
                # Table header
                print(f"{'Device':<8} {'Total':<12} {'Used':<12} {'Available':<12} {'Bar':<20}")
                print("-" * 70)
                
                for dev_id in pools:
                    stats = _get_pool_stats(dev_id)
                    if stats:
                        total = stats["total_bytes"]
                        used = stats["used_bytes"]
                        available = stats["available_bytes"]
                        bar = _format_bar(used, total, width=20)
                        
                        total_str = format_size(total)
                        used_str = format_size(used)
                        avail_str = format_size(available)
                        
                        print(f"{dev_id:<8} {total_str:<12} {used_str:<12} {avail_str:<12} [{bar}]")
                    else:
                        print(f"{dev_id:<8} {clr('Error reading stats', 'red')}")
            
            if not json_output:
                print()
                print(clr(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}", 'blue'))
            
            # Wait for next update
            time.sleep(interval)
    
    except KeyboardInterrupt:
        if not json_output:
            print()
            print(clr("Watch stopped.", 'yellow'))
        return 130


def cmd_top(device_id: Optional[int], refresh: float = 1.0) -> int:
    """
    Interactive TUI for monitoring (curses-based).
    
    Provides an interactive top-like interface for real-time monitoring
    of GPU memory pools.
    
    Args:
        device_id: Specific device to monitor, or None for all
        refresh: Refresh interval in seconds
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        import curses
    except ImportError:
        print_error("curses module not available. Try using 'faissctl watch' instead.")
        return 1
    
    def _draw_screen(stdscr):
        """Main curses draw loop."""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        
        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Header
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)   # OK
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)    # Critical
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal
        
        last_refresh = 0
        
        while True:
            current_time = time.time()
            
            # Handle input
            try:
                key = stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    last_refresh = 0  # Force refresh
            except:
                pass
            
            # Refresh if needed
            if current_time - last_refresh >= refresh:
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                
                # Header
                title = f"Faissctl Top - Refresh: {refresh:.1f}s (q:quit, r:refresh)"
                stdscr.addstr(0, 0, title[:width - 1], curses.color_pair(1) | curses.A_BOLD)
                
                if device_id is not None:
                    # Single device view
                    stats = _get_pool_stats(device_id)
                    if stats:
                        row = 2
                        stdscr.addstr(row, 0, f"GPU {device_id} Pool Statistics", curses.color_pair(1))
                        row += 2
                        
                        total = stats["total_bytes"]
                        used = stats["used_bytes"]
                        available = stats["available_bytes"]
                        util = stats["utilization"]
                        
                        stdscr.addstr(row, 0, f"Total:     {format_size(total)}")
                        row += 1
                        stdscr.addstr(row, 0, f"Used:      {format_size(used)} ({util * 100:.1f}%)")
                        row += 1
                        stdscr.addstr(row, 0, f"Available: {format_size(available)}")
                        row += 2
                        
                        # Bar
                        bar_width = min(50, width - 15)
                        if total > 0:
                            filled = int((used / total) * bar_width)
                            
                            if util < 0.5:
                                color = 2  # Green
                            elif util < 0.8:
                                color = 3  # Yellow
                            else:
                                color = 4  # Red
                            
                            bar = "█" * filled + "░" * (bar_width - filled)
                            stdscr.addstr(row, 0, f"[{bar}]", curses.color_pair(color))
                            stdscr.addstr(row, bar_width + 3, f"{util * 100:.1f}%")
                    else:
                        stdscr.addstr(2, 0, f"Error reading GPU {device_id} stats", curses.color_pair(4))
                else:
                    # Multi-device view
                    pools = detect_gpu_pools()
                    
                    row = 2
                    header = f"{'Device':<8} {'Total':<12} {'Used':<12} {'Available':<12} {'Util':<8}"
                    stdscr.addstr(row, 0, header, curses.color_pair(1) | curses.A_BOLD)
                    row += 1
                    stdscr.addstr(row, 0, "-" * min(60, width - 1), curses.color_pair(5))
                    row += 1
                    
                    for dev_id in pools:
                        if row >= height - 1:
                            break
                        
                        stats = _get_pool_stats(dev_id)
                        if stats:
                            total = stats["total_bytes"]
                            used = stats["used_bytes"]
                            available = stats["available_bytes"]
                            util = stats["utilization"]
                            
                            total_str = format_size(total)
                            used_str = format_size(used)
                            avail_str = format_size(available)
                            util_str = f"{util * 100:.1f}%"
                            
                            if util < 0.5:
                                color = 2
                            elif util < 0.8:
                                color = 3
                            else:
                                color = 4
                            
                            line = f"{dev_id:<8} {total_str:<12} {used_str:<12} {avail_str:<12} {util_str:<8}"
                            stdscr.addstr(row, 0, line[:width - 1], curses.color_pair(color))
                        else:
                            line = f"{dev_id:<8} Error reading stats"
                            stdscr.addstr(row, 0, line[:width - 1], curses.color_pair(4))
                        
                        row += 1
                
                # Footer
                footer = f"Last update: {time.strftime('%H:%M:%S')}"
                stdscr.addstr(height - 1, 0, footer, curses.color_pair(5))
                
                stdscr.refresh()
                last_refresh = current_time
            
            # Small sleep to prevent high CPU usage
            time.sleep(0.1)
    
    try:
        curses.wrapper(_draw_screen)
        return 0
    except Exception as e:
        print_error(f"TUI error: {e}")
        return 1
