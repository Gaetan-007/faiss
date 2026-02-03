"""
Main entry point for Faiss FastAPI server.
"""
import argparse
import sys
import uvicorn
from pathlib import Path

from .serve_config import ServerConfig
from .app import create_app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Faiss FastAPI Search Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level",
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON or YAML)",
    )
    
    # Engine config (can override config file)
    parser.add_argument(
        "--index-path",
        type=str,
        help="Path to Faiss index file",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        help="Path to corpus dataset",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        help="Number of probes for IVF index",
    )
    
    # Logging settings
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for log files",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file name",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="json",
        choices=["json", "text"],
        help="Log format",
    )
    parser.add_argument(
        "--no-request-logging",
        action="store_true",
        help="Disable request logging",
    )
    
    return parser.parse_args()


def load_config(args) -> ServerConfig:
    """Load configuration from file and/or command line arguments."""
    config_dict = {}
    
    # Load from config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        if config_path.suffix == ".json":
            import json
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Override with command line arguments
    if args.host:
        config_dict["host"] = args.host
    if args.port:
        config_dict["port"] = args.port
    if args.workers:
        config_dict["workers"] = args.workers
    if args.reload:
        config_dict["reload"] = True
    if args.log_level:
        config_dict["log_level"] = args.log_level
    if args.log_dir:
        config_dict["log_dir"] = args.log_dir
    if args.log_file:
        config_dict["log_file"] = args.log_file
    if args.log_format:
        config_dict["log_format"] = args.log_format
    if args.no_request_logging:
        config_dict["enable_request_logging"] = False
    
    # Override engine config
    engine_config = config_dict.get("engine_config", {})
    if args.index_path:
        engine_config["index_path"] = args.index_path
    if args.corpus_path:
        engine_config["corpus_path"] = args.corpus_path
    if args.nprobe:
        engine_config["nprobe"] = args.nprobe
    config_dict["engine_config"] = engine_config
    
    # Create config object
    try:
        config = ServerConfig.from_dict(config_dict, validate=False)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    # Validate required fields
    if config.engine_config:
        if not config.engine_config.get("index_path"):
            raise ValueError("index_path is required in engine_config")
        if not config.engine_config.get("corpus_path"):
            raise ValueError("corpus_path is required in engine_config")
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        config = load_config(args)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create FastAPI app
    app = create_app(config)
    
    # Run server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers if not config.reload else 1,
        reload=config.reload,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
