"""
Request logging middleware and utilities for Faiss server.
"""
import json
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestLogger:
    """Request logger that records all API requests and responses."""
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        log_format: str = "json",
        log_requests: bool = True,
        log_responses: bool = True,
        log_request_body: bool = True,
        log_response_body: bool = False,
        log_latency: bool = True,
    ):
        """
        Initialize request logger.
        
        Args:
            log_dir: Directory to store log files (if None, uses current directory)
            log_file: Specific log file name (if None, auto-generates)
            log_format: "json" or "text"
            log_requests: Whether to log requests
            log_responses: Whether to log responses
            log_request_body: Whether to log request body
            log_response_body: Whether to log response body
            log_latency: Whether to log request latency
        """
        self.log_format = log_format
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.log_latency = log_latency
        
        # Setup logging
        self.logger = logging.getLogger("faiss.server.request")
        self.logger.setLevel(logging.INFO)
        
        # Setup file handler if log_dir or log_file is specified
        if log_dir or log_file:
            if log_dir:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
            else:
                log_path = Path.cwd()
            
            if log_file:
                log_file_path = log_path / log_file
            else:
                # Auto-generate log file name with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = log_path / f"faiss_server_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            
            if log_format == "json":
                formatter = logging.Formatter('%(message)s')
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Also add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        if log_format == "json":
            console_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
        self.logger.addHandler(console_handler)
        
        # Statistics
        self.stats = defaultdict(int)
        self.stats["total_requests"] = 0
        self.stats["total_errors"] = 0
        self.stats["total_latency_ms"] = 0.0
        
    def log_request(
        self,
        request: Request,
        request_id: str,
        start_time: float,
        request_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log incoming request."""
        if not self.log_requests:
            return
        
        log_data = {
            "type": "request",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "headers": dict(request.headers),
        }
        
        if self.log_request_body and request_body:
            log_data["body"] = request_body
        
        if self.log_format == "json":
            self.logger.info(json.dumps(log_data, default=str))
        else:
            self.logger.info(
                f"REQUEST [{request_id}] {request.method} {request.url.path} "
                f"from {log_data['client_host']}"
            )
    
    def log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        start_time: float,
        response_body: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Log outgoing response."""
        if not self.log_responses:
            return
        
        latency_ms = (time.time() - start_time) * 1000
        
        log_data = {
            "type": "response",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        
        if self.log_latency:
            log_data["latency_ms"] = round(latency_ms, 2)
        
        if error:
            log_data["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            self.stats["total_errors"] += 1
        else:
            self.stats["total_requests"] += 1
            self.stats["total_latency_ms"] += latency_ms
        
        if self.log_response_body and response_body:
            # Truncate large response bodies
            if isinstance(response_body, (dict, list)):
                body_str = json.dumps(response_body, default=str)
                if len(body_str) > 10000:  # 10KB limit
                    log_data["body"] = body_str[:10000] + "... (truncated)"
                else:
                    log_data["body"] = response_body
            else:
                log_data["body"] = str(response_body)[:10000]
        
        if self.log_format == "json":
            self.logger.info(json.dumps(log_data, default=str))
        else:
            status_emoji = "✅" if response.status_code < 400 else "❌"
            self.logger.info(
                f"RESPONSE [{request_id}] {status_emoji} {response.status_code} "
                f"{request.method} {request.url.path} "
                f"({latency_ms:.2f}ms)"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        total = self.stats["total_requests"]
        avg_latency = (
            self.stats["total_latency_ms"] / total
            if total > 0
            else 0.0
        )
        return {
            "total_requests": total,
            "total_errors": self.stats["total_errors"],
            "average_latency_ms": round(avg_latency, 2),
        }


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for request/response logging."""
    
    def __init__(
        self,
        app: ASGIApp,
        logger: RequestLogger,
    ):
        super().__init__(app)
        self.logger = logger
        self.request_id_counter = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log it."""
        import uuid
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Read request body if needed
        request_body = None
        if self.logger.log_request_body:
            try:
                body = await request.body()
                if body:
                    try:
                        request_body = json.loads(body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        request_body = {"raw": body[:1000].decode(errors="ignore")}
            except Exception:
                pass
        
        # Log request
        self.logger.log_request(request, request_id, start_time, request_body)
        
        # Process request
        response = None
        error = None
        try:
            response = await call_next(request)
        except Exception as e:
            error = e
            # Create error response
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
            raise
        finally:
            # Log response
            if response:
                response_body = None
                if self.logger.log_response_body:
                    # Note: Reading response body here is tricky with FastAPI
                    # For now, we'll skip it or implement a custom response class
                    pass
                
                self.logger.log_response(
                    request, response, request_id, start_time, response_body, error
                )
        
        return response
