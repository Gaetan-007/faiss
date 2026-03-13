"""
FastAPI application for Faiss search server.
"""
import time
import uuid
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .logger import RequestLogger, RequestLoggingMiddleware
from .serve_config import ServerConfig
from ..engine.engine import FaissEngine, FaissEnginConfig, EngineError
from ..engine import BaseEngine
from ..utils import BaseConfig


# Global engine instance
_engine: Optional[BaseEngine] = None
_config: Optional[ServerConfig] = None
_logger: Optional[RequestLogger] = None


class _UnavailableEngine(BaseEngine):
    """Engine placeholder used when initialization fails."""

    def __init__(self, message: str):
        super().__init__(BaseConfig())
        self._message = message

    def _search(self, query: str, num: int, return_score: bool):
        raise EngineError(self._message)

    def _batch_search(self, query_list, query_id_list, num, return_score, eval_cache):
        raise EngineError(self._message)

    def _add(self, query: str, return_centroid: bool, retrain: bool):
        raise EngineError(self._message)

    def _batch_add(self, query_list, return_centroid: bool, retrain: bool):
        raise EngineError(self._message)


def get_engine() -> BaseEngine:
    """Dependency to get engine instance."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


def get_logger() -> RequestLogger:
    """Dependency to get logger instance."""
    if _logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialized")
    return _logger


def create_app(config: ServerConfig) -> FastAPI:
    """
    Create and return FastAPI application.
    
    Args:
        config: Server configuration
        
    Returns:
        FastAPI application instance
    """
    global _config, _logger, _engine
    
    _config = config
    _engine = None
    if config.enable_request_logging:
        _logger = RequestLogger(
            log_dir=config.log_dir,
            log_file=config.log_file,
            log_format=config.log_format,
            log_requests=config.log_requests,
            log_responses=config.log_responses,
            log_request_body=config.log_request_body,
            log_response_body=config.log_response_body,
            log_latency=config.log_latency,
        )
    else:
        _logger = None
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup/shutdown."""
        global _engine, _logger
        
        # Startup
        # Initialize logger if not already set
        if _logger is None and config.enable_request_logging:
            _logger = RequestLogger(
                log_dir=config.log_dir,
                log_file=config.log_file,
                log_format=config.log_format,
                log_requests=config.log_requests,
                log_responses=config.log_responses,
                log_request_body=config.log_request_body,
                log_response_body=config.log_response_body,
                log_latency=config.log_latency,
            )
        
        # Initialize engine
        engine_error = None
        if not config.engine_config:
            engine_error = "engine_config is empty; engine initialization skipped"
        else:
            missing_keys = [
                key for key in ("index_path", "corpus_path")
                if not config.engine_config.get(key)
            ]
            if missing_keys:
                engine_error = (
                    "engine_config missing required fields: "
                    + ", ".join(missing_keys)
                )

        if engine_error:
            _engine = _UnavailableEngine(engine_error)
            app.state.engine_error = engine_error
        else:
            try:
                engine_config = FaissEnginConfig.from_dict(config.engine_config)
                _engine = FaissEngine(engine_config)
            except Exception as exc:
                engine_error = f"engine initialization failed: {exc}"
                _engine = _UnavailableEngine(engine_error)
                app.state.engine_error = engine_error
        
        app.state.engine = _engine
        app.state.logger = _logger
        app.state.config = config
        
        yield
        
        # Shutdown
        if _engine and hasattr(_engine, "shutdown_async"):
            _engine.shutdown_async(drain=True)
    
    # Create FastAPI app
    app = FastAPI(
        title="Faiss Search Server",
        description="FastAPI server for Faiss-based similarity search",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add request logging middleware
    # We'll add it dynamically after logger is initialized
    # For now, we'll use a simpler approach with a custom middleware wrapper
    if config.enable_request_logging:
        # We'll add the middleware in the lifespan after logger is created
        # This requires a different approach - we'll use a wrapper
        pass
    
    # Include routers
    from .routes import router
    app.include_router(router, prefix=config.api_prefix)
    
    app.state.logger = _logger

    # Add middleware wrapper that checks for logger
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Request logging middleware."""
        active_logger = getattr(request.app.state, "logger", None) or _logger
        if active_logger and config.enable_request_logging:
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Read request body if needed
            request_body = None
            if active_logger.log_request_body:
                try:
                    body = await request.body()
                    if body:
                        try:
                            import json
                            request_body = json.loads(body.decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_body = {"raw": body[:1000].decode(errors="ignore")}
                except Exception:
                    pass
            
            # Log request
            active_logger.log_request(request, request_id, start_time, request_body)
            
            # Process request
            response = None
            error = None
            try:
                response = await call_next(request)
            except Exception as e:
                error = e
                from fastapi.responses import JSONResponse
                response = JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        }
                    }
                )
            
            # Log response
            if response:
                active_logger.log_response(
                    request, response, request_id, start_time, None, error
                )
            
            return response
        else:
            return await call_next(request)
    
    return app
