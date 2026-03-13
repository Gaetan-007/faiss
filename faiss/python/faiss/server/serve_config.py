"""
Server configuration for Faiss FastAPI server.
"""
import json
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

from ..utils import BaseConfig


@dataclass
class ServerConfig(BaseConfig):
    """Configuration for Faiss FastAPI server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    
    # Engine configuration (will be passed to FaissEngine)
    engine_config: Dict[str, Any] = field(default_factory=dict)
    
    # Logging configuration
    enable_request_logging: bool = True
    log_dir: Optional[str] = None
    log_file: Optional[str] = None
    log_format: str = "json"  # "json" or "text"
    log_requests: bool = True
    log_responses: bool = True
    log_request_body: bool = True
    log_response_body: bool = False  # Can be large, disabled by default
    log_latency: bool = True
    
    # API settings
    api_prefix: str = "/api/v1"
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # Health check settings
    health_check_path: str = "/health"
    ready_check_path: str = "/ready"
    
    # Rate limiting (optional)
    enable_rate_limit: bool = False
    rate_limit_per_minute: int = 1000
    
    @classmethod
    def from_dict(cls, config: dict, *, validate: bool = True):
        """Create config from dictionary."""
        # Handle nested engine_config
        engine_config = config.pop("engine_config", {})
        if not isinstance(engine_config, dict):
            raise ValueError("engine_config must be a dict")
        if validate:
            required_keys = ("index_path", "corpus_path")
            missing = [key for key in required_keys if not engine_config.get(key)]
            if missing:
                raise ValueError(
                    "engine_config missing required fields: "
                    + ", ".join(missing)
                )
        instance = cls(**config)
        instance.engine_config = engine_config
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "reload": self.reload,
            "log_level": self.log_level,
            "engine_config": self.engine_config,
            "enable_request_logging": self.enable_request_logging,
            "log_dir": self.log_dir,
            "log_file": self.log_file,
            "log_format": self.log_format,
            "log_requests": self.log_requests,
            "log_responses": self.log_responses,
            "log_request_body": self.log_request_body,
            "log_response_body": self.log_response_body,
            "log_latency": self.log_latency,
            "api_prefix": self.api_prefix,
            "enable_cors": self.enable_cors,
            "cors_origins": self.cors_origins,
            "max_request_size": self.max_request_size,
            "health_check_path": self.health_check_path,
            "ready_check_path": self.ready_check_path,
            "enable_rate_limit": self.enable_rate_limit,
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }
