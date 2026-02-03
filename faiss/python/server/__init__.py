"""
Faiss FastAPI Server module.

This module provides a FastAPI-based server for Faiss search operations
with request logging capabilities.
"""
from .app import create_app, get_engine, get_logger
from .serve_config import ServerConfig
from .logger import RequestLogger, RequestLoggingMiddleware
from .routes import router
from .main import main

__all__ = [
    "create_app",
    "get_engine",
    "get_logger",
    "ServerConfig",
    "RequestLogger",
    "RequestLoggingMiddleware",
    "router",
    "main",
]
