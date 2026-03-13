# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Faissctl - Unified CLI for Faiss IPC management.

Entry point for the faissctl command-line tool.
"""

import argparse
import sys
from typing import List, Optional

from . import __version__
from .utils import clr, print_error


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="faissctl",
        description="Unified CLI for Faiss GPU memory pool and IVF list management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  faissctl pool list                     # List all GPU memory pools
  faissctl pool stats 0                  # Show GPU 0 pool statistics
  faissctl pool expand 0 4G            # Expand GPU 0 pool to 4GB
  faissctl pool shrink 0 2G             # Shrink GPU 0 pool to 2GB
  faissctl ivf status /path/to/index     # Show evicted IVF lists
  faissctl ivf evict /path/to/index 42   # Evict list 42
  faissctl ivf load /path/to/index 42    # Load list 42
  faissctl watch -n 2 0                  # Watch GPU 0 pool every 2 seconds
  faissctl top                           # Interactive TUI mode
        """,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format where applicable",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pool commands
    pool_parser = subparsers.add_parser(
        "pool",
        help="GPU memory pool management",
        description="Manage GPU memory pools via shared memory IPC",
    )
    pool_subparsers = pool_parser.add_subparsers(dest="pool_cmd", help="Pool commands")
    
    # pool list
    pool_list_parser = pool_subparsers.add_parser(
        "list",
        help="List all GPU memory pools",
    )
    
    # pool stats
    pool_stats_parser = pool_subparsers.add_parser(
        "stats",
        help="Show pool statistics",
    )
    pool_stats_parser.add_argument(
        "device_id",
        type=int,
        help="GPU device ID",
    )
    
    # pool expand
    pool_expand_parser = pool_subparsers.add_parser(
        "expand",
        help="Expand pool to target size",
    )
    pool_expand_parser.add_argument(
        "device_id",
        type=int,
        help="GPU device ID",
    )
    pool_expand_parser.add_argument(
        "target_size",
        type=str,
        help="Target size (e.g., 2G, 512M, 1T)",
    )
    pool_expand_parser.add_argument(
        "--timeout",
        type=int,
        default=5000,
        metavar="MS",
        help="Timeout in milliseconds (default: 5000)",
    )
    
    # pool shrink
    pool_shrink_parser = pool_subparsers.add_parser(
        "shrink",
        help="Shrink pool to target size",
    )
    pool_shrink_parser.add_argument(
        "device_id",
        type=int,
        help="GPU device ID",
    )
    pool_shrink_parser.add_argument(
        "target_size",
        type=str,
        help="Target size (e.g., 1G, 256M)",
    )
    pool_shrink_parser.add_argument(
        "--timeout",
        type=int,
        default=5000,
        metavar="MS",
        help="Timeout in milliseconds (default: 5000)",
    )
    
    # IVF commands
    ivf_parser = subparsers.add_parser(
        "ivf",
        help="IVF list management",
        description="Manage IVF lists on GPU indices",
    )
    ivf_subparsers = ivf_parser.add_subparsers(dest="ivf_cmd", help="IVF commands")
    
    # ivf status
    ivf_status_parser = ivf_subparsers.add_parser(
        "status",
        help="Show evicted IVF lists",
    )
    ivf_status_parser.add_argument(
        "index_path",
        type=str,
        help="Path to Faiss index file",
    )
    
    # ivf evict
    ivf_evict_parser = ivf_subparsers.add_parser(
        "evict",
        help="Evict IVF list to CPU",
    )
    ivf_evict_parser.add_argument(
        "index_path",
        type=str,
        help="Path to Faiss index file",
    )
    ivf_evict_parser.add_argument(
        "list_id",
        type=int,
        help="IVF list ID to evict",
    )
    
    # ivf load
    ivf_load_parser = ivf_subparsers.add_parser(
        "load",
        help="Load IVF list to GPU",
    )
    ivf_load_parser.add_argument(
        "index_path",
        type=str,
        help="Path to Faiss index file",
    )
    ivf_load_parser.add_argument(
        "list_id",
        type=int,
        help="IVF list ID to load",
    )
    
    # Watch command
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch pool statistics continuously",
        description="Continuously monitor and display pool statistics",
    )
    watch_parser.add_argument(
        "-n",
        "--interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Update interval in seconds (default: 1.0)",
    )
    watch_parser.add_argument(
        "device_id",
        nargs="?",
        type=int,
        default=None,
        help="GPU device ID (default: all devices)",
    )
    
    # Top command
    top_parser = subparsers.add_parser(
        "top",
        help="Interactive TUI for monitoring",
        description="Launch interactive terminal UI for real-time monitoring",
    )
    top_parser.add_argument(
        "-r",
        "--refresh",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Refresh interval in seconds (default: 1.0)",
    )
    top_parser.add_argument(
        "device_id",
        nargs="?",
        type=int,
        default=None,
        help="GPU device ID (default: all devices)",
    )
    
    return parser


def handle_pool_command(args) -> int:
    """Handle pool subcommands."""
    from .pool_commands import (
        cmd_pool_expand,
        cmd_pool_list,
        cmd_pool_shrink,
        cmd_pool_stats,
    )
    
    if args.pool_cmd == "list" or args.pool_cmd is None:
        return cmd_pool_list(json_output=args.json)
    elif args.pool_cmd == "stats":
        return cmd_pool_stats(args.device_id, json_output=args.json)
    elif args.pool_cmd == "expand":
        return cmd_pool_expand(args.device_id, args.target_size, timeout_ms=args.timeout)
    elif args.pool_cmd == "shrink":
        return cmd_pool_shrink(args.device_id, args.target_size, timeout_ms=args.timeout)
    else:
        print_error(f"Unknown pool command: {args.pool_cmd}")
        return 1


def handle_ivf_command(args) -> int:
    """Handle IVF subcommands."""
    from .ivf_commands import cmd_ivf_evict, cmd_ivf_load, cmd_ivf_status
    
    if args.ivf_cmd == "status" or args.ivf_cmd is None:
        return cmd_ivf_status(args.index_path, json_output=args.json)
    elif args.ivf_cmd == "evict":
        return cmd_ivf_evict(args.index_path, args.list_id)
    elif args.ivf_cmd == "load":
        return cmd_ivf_load(args.index_path, args.list_id)
    else:
        print_error(f"Unknown IVF command: {args.ivf_cmd}")
        return 1


def handle_watch_command(args) -> int:
    """Handle watch command."""
    from .tui import cmd_watch
    
    return cmd_watch(args.device_id, interval=args.interval, json_output=args.json)


def handle_top_command(args) -> int:
    """Handle top command."""
    from .tui import cmd_top
    
    return cmd_top(args.device_id, refresh=args.refresh)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for faissctl.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command is None:
        parser.print_help()
        return 0
    
    try:
        if parsed_args.command == "pool":
            return handle_pool_command(parsed_args)
        elif parsed_args.command == "ivf":
            return handle_ivf_command(parsed_args)
        elif parsed_args.command == "watch":
            return handle_watch_command(parsed_args)
        elif parsed_args.command == "top":
            return handle_top_command(parsed_args)
        else:
            print_error(f"Unknown command: {parsed_args.command}")
            return 1
    except KeyboardInterrupt:
        print()
        print_error("Interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
