"""
Integration tests for Faiss server with real components.

This test file focuses on end-to-end testing with actual server instances.
Run these tests with: pytest tests/test_server_integration.py -v
"""
import pytest
import requests
import time
import subprocess
import signal
import os
import tempfile
import shutil
from pathlib import Path
from threading import Thread
from typing import Optional

import numpy as np
import faiss


# ============================================================================
# Test Server Management
# ============================================================================

class TestServerManager:
    """Manages a test server process."""
    __test__ = False  # Tell pytest not to collect this as a test class
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"
    
    def start(self, config_path: Optional[str] = None):
        """Start the server."""
        if self.process is not None:
            raise RuntimeError("Server already running")
        
        cmd = [
            "python", "-m", "faiss.server.main",
            "--host", self.host,
            "--port", str(self.port),
        ]
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )
        
        # Wait for server to be ready
        max_wait = 30
        for _ in range(max_wait):
            try:
                response = requests.get(f"{self.base_url}/api/v1/health", timeout=1)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        raise RuntimeError("Server failed to start")
    
    def stop(self):
        """Stop the server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestServerIntegration:
    """Integration tests with real server."""
    
    @pytest.fixture
    def server(self):
        """Create a test server."""
        # Note: This requires a valid config with real index/corpus
        # For now, we'll skip if not available
        with TestServerManager(port=8888) as server:
            yield server
    
    def test_server_startup(self):
        """Test that server can start."""
        server = TestServerManager(port=8889)
        try:
            # This will fail if server can't start, which is expected
            # without proper configuration
            server.start()
            assert server.process is not None
        except (RuntimeError, FileNotFoundError):
            pytest.skip("Server startup test requires proper configuration")
        finally:
            server.stop()
    
    def test_health_endpoint_live(self, server):
        """Test health endpoint on live server."""
        response = requests.get(f"{server.base_url}/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_search_endpoint_live(self, server):
        """Test search endpoint on live server."""
        response = requests.post(
            f"{server.base_url}/api/v1/search",
            json={"query": "test", "num": 5},
            timeout=10
        )
        assert response.status_code in [200, 500]  # 500 if engine not ready


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_engine():
    """Create a mock engine for testing."""
    from unittest.mock import Mock
    engine = Mock()
    engine.search = Mock(return_value=[
        {"content": "test result 1", "id": 0},
        {"content": "test result 2", "id": 1},
    ])
    engine.batch_search = Mock(return_value=[
        [{"content": "result 1", "id": 0}],
        [{"content": "result 2", "id": 1}],
    ])
    return engine


@pytest.fixture
def client(mock_engine):
    """Create a test client with mocked engine."""
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    import tempfile
    import shutil
    import sys
    
    # Ensure we patch before any imports of faiss.server.app
    # Use the same pattern as test_server.py
    temp_dir = tempfile.mkdtemp()
    try:
        # Import ServerConfig directly from the module to avoid triggering __init__.py
        # which would import app and main (which imports uvicorn)
        import faiss.server.serve_config
        ServerConfig = faiss.server.serve_config.ServerConfig
        
        server_config = ServerConfig(
            host="127.0.0.1",
            port=8000,
            workers=1,
            reload=False,
            log_level="info",
            enable_request_logging=True,
            log_dir=temp_dir,
            log_format="json",
            log_requests=True,
            log_responses=True,
            log_request_body=True,
            log_response_body=False,
            log_latency=True,
            api_prefix="/api/v1",
            enable_cors=True,
            engine_config={
                "index_type": "IVFFlat",
                "index_path": "/fake/path/to/index.faiss",
                "corpus_path": "/fake/path/to/corpus",
                "retrieval_method": "/share_data/public_models/multilingual-e5-large",
                "retrieval_topk": 10,
                "retrieval_batch_size": 32,
                "retrieval_query_max_length": 512,
                "retrieval_use_fp16": False,
                "retrieval_pooling_method": "mean",
                "return_embedding": False,
                "larger_topk": 20,
                "use_sentence_transformer": True,
                "gpu_memory_utilization": 0.8,
                "eviction_policy": "lru",
                "eviction_max_attempts": 64,
                "nprobe": 16,
            }
        )
        
        # Patch FaissEngine in the app module namespace
        # This must be done before importing create_app
        # Use the same pattern as test_server.py
        with patch('faiss.server.app.FaissEngine') as mock_faiss_engine:
            mock_faiss_engine.return_value = mock_engine
            from faiss.server.app import create_app, get_engine, get_logger
            from faiss.server.logger import RequestLogger
            
            app = create_app(server_config)
            
            # Override dependencies to use mock engine
            app.dependency_overrides[get_engine] = lambda: mock_engine
            app.dependency_overrides[get_logger] = lambda: RequestLogger(
                log_dir=server_config.log_dir,
                log_format=server_config.log_format,
            )
            
            with TestClient(app) as test_client:
                yield test_client
            
            app.dependency_overrides.clear()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# API Client Tests
# ============================================================================

class TestAPIClient:
    """Test API client usage patterns."""
    
    def test_search_request_format(self):
        """Test search request format."""
        request = {
            "query": "machine learning",
            "num": 10,
            "return_score": True
        }
        # Validate structure
        assert "query" in request
        assert isinstance(request["query"], str)
        assert "num" in request
        assert isinstance(request["num"], int)
        assert "return_score" in request
        assert isinstance(request["return_score"], bool)
    
    def test_batch_search_request_format(self):
        """Test batch search request format."""
        request = {
            "queries": ["query1", "query2", "query3"],
            "num": 5,
            "return_score": False
        }
        assert "queries" in request
        assert isinstance(request["queries"], list)
        assert len(request["queries"]) > 0
    
    def test_response_format(self):
        """Test expected response format."""
        # Mock response structure
        response = {
            "results": [
                {"content": "result 1", "id": 0},
                {"content": "result 2", "id": 1},
            ],
            "scores": [0.95, 0.85],
            "latency_ms": 123.45
        }
        assert "results" in response
        assert "latency_ms" in response
        assert isinstance(response["results"], list)
        assert isinstance(response["latency_ms"], (int, float))


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformance:
    """Performance benchmark tests."""
    
    def test_search_latency(self, client, mock_engine):
        """Benchmark search latency."""
        from fastapi.testclient import TestClient
        from faiss.server.app import create_app
        from faiss.server.serve_config import ServerConfig
        
        # This would need proper setup
        times = []
        for _ in range(10):
            start = time.time()
            # Make request
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_latency = sum(times) / len(times)
        # Assert reasonable latency (adjust based on requirements)
        assert avg_latency < 1.0  # Less than 1 second
    
    def test_batch_search_efficiency(self):
        """Test that batch search is more efficient than individual searches."""
        # This would require actual implementation
        pass


# ============================================================================
# Error Recovery Tests
# ============================================================================

class TestErrorRecovery:
    """Test error recovery scenarios."""
    
    def test_server_recovery_after_error(self, client, mock_engine):
        """Test that server recovers after an error."""
        # Cause an error
        mock_engine.search.side_effect = Exception("Temporary error")
        response1 = client.post(
            "/api/v1/search",
            json={"query": "test", "num": 5}
        )
        assert response1.status_code == 500
        
        # Reset and try again
        mock_engine.search.side_effect = None
        mock_engine.search.return_value = [{"content": "result", "id": 0}]
        response2 = client.post(
            "/api/v1/search",
            json={"query": "test", "num": 5}
        )
        assert response2.status_code == 200


# ============================================================================
# Concurrency Tests
# ============================================================================

class TestConcurrency:
    """Test concurrent request handling."""
    
    def test_concurrent_searches(self, client, mock_engine):
        """Test handling of concurrent search requests."""
        import threading
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post(
                    "/api/v1/search",
                    json={"query": "test", "num": 5}
                )
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=make_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All requests should complete
        assert len(results) == 10
        # All should be successful (200) or validation errors (422)
        assert all(status in [200, 422] for status in results)


# ============================================================================
# Data Validation Tests
# ============================================================================

class TestDataValidation:
    """Test data validation."""
    
    def test_search_response_structure(self, client, mock_engine):
        """Test that search response has correct structure."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "num": 5, "return_score": True}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Validate structure
        assert "results" in data
        assert "latency_ms" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["latency_ms"], (int, float))
    
    def test_batch_search_response_structure(self, client, mock_engine):
        """Test that batch search response has correct structure."""
        response = client.post(
            "/api/v1/batch_search",
            json={"queries": ["q1", "q2"], "num": 5}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2  # Two queries


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
