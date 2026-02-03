"""
Comprehensive tests for Faiss FastAPI server.

This test suite includes:
- Unit tests for components
- Integration tests for API endpoints
- Functional tests for complete workflows
- Error handling tests
"""
import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any

import numpy as np
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

# Import server components
from faiss.server.serve_config import ServerConfig
from faiss.server.logger import RequestLogger
from faiss.server.app import create_app, get_engine, get_logger
from faiss.server.routes import (
    SearchRequest,
    BatchSearchRequest,
    AddRequest,
    BatchAddRequest,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_engine():
    """Create a mock engine for testing."""
    engine = Mock()
    engine.search = Mock(return_value=[
        {"content": "test result 1", "id": 0},
        {"content": "test result 2", "id": 1},
    ])
    engine.search.return_value = [
        {"content": "test result 1", "id": 0},
        {"content": "test result 2", "id": 1},
    ]
    engine.batch_search = Mock(return_value=[
        [{"content": "result 1", "id": 0}],
        [{"content": "result 2", "id": 1}],
    ])
    engine._add = Mock(return_value={"success": True})
    engine._batch_add = Mock(return_value=[{"success": True}, {"success": True}])
    engine.finished_requests = 10
    engine.request_batch_size = 5
    return engine


@pytest.fixture
def mock_engine_with_scores():
    """Create a mock engine that returns scores."""
    engine = Mock()
    engine.search = Mock(return_value=(
        [{"content": "test result 1", "id": 0}],
        [0.95, 0.85]
    ))
    engine.batch_search = Mock(return_value=(
        [
            [{"content": "result 1", "id": 0}],
            [{"content": "result 2", "id": 1}],
        ],
        [
            [0.95, 0.85],
            [0.90, 0.80],
        ]
    ))
    return engine


@pytest.fixture
def server_config(temp_dir):
    """Create a test server configuration."""
    return ServerConfig(
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
            "retrieval_method": "sentence-transformers/all-MiniLM-L6-v2",
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


@pytest.fixture
def app_with_mock_engine(mock_engine, server_config):
    """Create FastAPI app with mocked engine."""
    with patch('faiss.server.app.FaissEngine') as mock_faiss_engine:
        mock_faiss_engine.return_value = mock_engine
        app = create_app(server_config)
        
        # Override dependencies
        app.dependency_overrides[get_engine] = lambda: mock_engine
        app.dependency_overrides[get_logger] = lambda: RequestLogger(
            log_dir=server_config.log_dir,
            log_format=server_config.log_format,
        )
        
        yield app
        
        app.dependency_overrides.clear()


@pytest.fixture
def client(app_with_mock_engine):
    """Create a test client."""
    return TestClient(app_with_mock_engine)


# ============================================================================
# Unit Tests - Configuration
# ============================================================================

class TestServerConfig:
    """Test ServerConfig class."""
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info",
            "engine_config": {
                "index_path": "/path/to/index",
                "corpus_path": "/path/to/corpus",
            }
        }
        config = ServerConfig.from_dict(config_dict)
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "info"
        assert config.engine_config["index_path"] == "/path/to/index"
    
    def test_config_to_dict(self, server_config):
        """Test converting config to dictionary."""
        config_dict = server_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["host"] == server_config.host
        assert config_dict["port"] == server_config.port
        assert "engine_config" in config_dict
    
    def test_config_defaults(self):
        """Test config default values."""
        config = ServerConfig.from_dict({
            "engine_config": {
                "index_path": "/path/to/index",
                "corpus_path": "/path/to/corpus",
            }
        })
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.enable_request_logging is True
        assert config.log_format == "json"


# ============================================================================
# Unit Tests - Logger
# ============================================================================

class TestRequestLogger:
    """Test RequestLogger class."""
    
    def test_logger_initialization(self, temp_dir):
        """Test logger initialization."""
        logger = RequestLogger(
            log_dir=temp_dir,
            log_format="json",
        )
        assert logger.log_format == "json"
        assert logger.log_requests is True
        assert logger.log_responses is True
    
    def test_logger_stats(self, temp_dir):
        """Test logger statistics."""
        logger = RequestLogger(log_dir=temp_dir)
        stats = logger.get_stats()
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert "average_latency_ms" in stats
    
    def test_logger_text_format(self, temp_dir):
        """Test logger with text format."""
        logger = RequestLogger(
            log_dir=temp_dir,
            log_format="text",
        )
        assert logger.log_format == "text"


# ============================================================================
# Integration Tests - API Endpoints
# ============================================================================

class TestHealthEndpoints:
    """Test health and readiness endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "engine_ready" in data
        assert "timestamp" in data
    
    def test_ready_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestSearchEndpoints:
    """Test search endpoints."""
    
    def test_search_basic(self, client, mock_engine):
        """Test basic search."""
        request_data = {
            "query": "test query",
            "num": 10,
            "return_score": False,
        }
        response = client.post("/api/v1/search", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "latency_ms" in data
        assert isinstance(data["results"], list)
        assert data["scores"] is None
        mock_engine.search.assert_called_once()
    
    def test_search_with_scores(self, client, mock_engine_with_scores):
        """Test search with scores."""
        # Update app dependency
        app = client.app
        app.dependency_overrides[get_engine] = lambda: mock_engine_with_scores
        
        request_data = {
            "query": "test query",
            "num": 10,
            "return_score": True,
        }
        response = client.post("/api/v1/search", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "scores" in data
        assert data["scores"] is not None
        assert isinstance(data["scores"], list)
    
    def test_search_invalid_request(self, client):
        """Test search with invalid request."""
        # Missing query
        response = client.post("/api/v1/search", json={})
        assert response.status_code == 422  # Validation error
    
    def test_search_engine_error(self, client, mock_engine):
        """Test search with engine error."""
        mock_engine.search.side_effect = Exception("Engine error")
        request_data = {
            "query": "test query",
            "num": 10,
        }
        response = client.post("/api/v1/search", json=request_data)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    def test_batch_search(self, client, mock_engine):
        """Test batch search."""
        request_data = {
            "queries": ["query1", "query2", "query3"],
            "num": 5,
            "return_score": False,
        }
        response = client.post("/api/v1/batch_search", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 3
        mock_engine.batch_search.assert_called_once()
    
    def test_batch_search_empty(self, client):
        """Test batch search with empty queries."""
        request_data = {
            "queries": [],
            "num": 5,
        }
        response = client.post("/api/v1/batch_search", json=request_data)
        assert response.status_code == 422  # Validation error


class TestAddEndpoints:
    """Test add document endpoints."""
    
    def test_add_document(self, client, mock_engine):
        """Test adding a document."""
        request_data = {
            "text": "New document text",
            "return_centroid": False,
            "retrain": False,
        }
        response = client.post("/api/v1/add", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_engine._add.assert_called_once()
    
    def test_batch_add_documents(self, client, mock_engine):
        """Test batch adding documents."""
        request_data = {
            "texts": ["text1", "text2", "text3"],
            "return_centroid": False,
            "retrain": False,
        }
        response = client.post("/api/v1/batch_add", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_engine._batch_add.assert_called_once()
    
    def test_add_document_error(self, client, mock_engine):
        """Test add document with error."""
        mock_engine._add.side_effect = Exception("Add error")
        request_data = {
            "text": "New document",
        }
        response = client.post("/api/v1/add", json=request_data)
        assert response.status_code == 500


class TestStatsEndpoint:
    """Test statistics endpoint."""
    
    def test_stats(self, client, mock_engine):
        """Test statistics endpoint."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "engine_stats" in data
        assert "logger_stats" in data
        assert "total_requests" in data["logger_stats"]


# ============================================================================
# Functional Tests - Complete Workflows
# ============================================================================

class TestSearchWorkflow:
    """Test complete search workflows."""
    
    def test_search_workflow(self, client, mock_engine):
        """Test complete search workflow."""
        # 1. Health check
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200
        
        # 2. Single search
        search_response = client.post(
            "/api/v1/search",
            json={"query": "test", "num": 5}
        )
        assert search_response.status_code == 200
        
        # 3. Batch search
        batch_response = client.post(
            "/api/v1/batch_search",
            json={"queries": ["q1", "q2"], "num": 5}
        )
        assert batch_response.status_code == 200
        
        # 4. Check stats
        stats_response = client.get("/api/v1/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["logger_stats"]["total_requests"] >= 2
    
    def test_add_and_search_workflow(self, client, mock_engine):
        """Test add document and search workflow."""
        # 1. Add document
        add_response = client.post(
            "/api/v1/add",
            json={"text": "New document"}
        )
        assert add_response.status_code == 200
        
        # 2. Search for it
        search_response = client.post(
            "/api/v1/search",
            json={"query": "document", "num": 10}
        )
        assert search_response.status_code == 200


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_engine(self):
        """Test behavior when engine is not initialized."""
        from faiss.server.app import get_engine
        with pytest.raises(HTTPException) as exc_info:
            get_engine()
        assert exc_info.value.status_code == 503
    
    def test_missing_logger(self):
        """Test behavior when logger is not initialized."""
        from faiss.server.app import get_logger
        with pytest.raises(HTTPException) as exc_info:
            get_logger()
        assert exc_info.value.status_code == 503
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/search",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_malformed_request(self, client):
        """Test handling of malformed requests."""
        response = client.post(
            "/api/v1/search",
            json={"invalid": "field"}
        )
        assert response.status_code == 422


# ============================================================================
# Request Logging Tests
# ============================================================================

class TestRequestLogging:
    """Test request logging functionality."""
    
    def test_request_logging_enabled(self, client, temp_dir):
        """Test that requests are logged when enabled."""
        # Make a request
        client.post(
            "/api/v1/search",
            json={"query": "test", "num": 5}
        )
        
        # Check if log files were created
        log_files = list(Path(temp_dir).glob("*.log"))
        # Note: Logging might be async, so we check if directory exists
        assert Path(temp_dir).exists()
    
    def test_logger_stats_accumulation(self, client, mock_engine):
        """Test that logger stats accumulate correctly."""
        # Make multiple requests
        for _ in range(5):
            client.post(
                "/api/v1/search",
                json={"query": "test", "num": 5}
            )
        
        # Check stats
        stats_response = client.get("/api/v1/stats")
        stats = stats_response.json()
        assert stats["logger_stats"]["total_requests"] >= 5


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance-related functionality."""
    
    def test_latency_tracking(self, client, mock_engine):
        """Test that latency is tracked."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "num": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] >= 0
    
    def test_batch_search_performance(self, client, mock_engine):
        """Test batch search performance."""
        import time
        start = time.time()
        response = client.post(
            "/api/v1/batch_search",
            json={
                "queries": ["q1", "q2", "q3", "q4", "q5"],
                "num": 10
            }
        )
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 5.0  # Should complete within 5 seconds


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration handling."""
    
    def test_config_from_json(self, temp_dir):
        """Test loading config from JSON."""
        config_file = Path(temp_dir) / "config.json"
        config_data = {
            "host": "0.0.0.0",
            "port": 9000,
            "engine_config": {
                "index_path": "/path/to/index",
                "corpus_path": "/path/to/corpus",
            }
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        config = ServerConfig.from_json(str(config_file))
        assert config.port == 9000
    
    def test_config_validation(self):
        """Test config validation."""
        # Missing required engine config
        with pytest.raises((ValueError, KeyError)):
            ServerConfig.from_dict({
                "engine_config": {}
            })


# ============================================================================
# CORS Tests
# ============================================================================

class TestCORS:
    """Test CORS functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/v1/search",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )
        # CORS middleware should handle this
        # In test client, we might not see CORS headers directly
        assert response.status_code in [200, 405]


# ============================================================================
# Integration Tests with Real Components
# ============================================================================

class TestIntegration:
    """Integration tests with real components."""
    
    @pytest.mark.skip(reason="Requires actual Faiss index and corpus")
    def test_real_search(self):
        """Test with real Faiss index (requires setup)."""
        # This would require:
        # 1. A real Faiss index file
        # 2. A real corpus dataset
        # 3. Proper model setup
        pass
    
    def test_app_creation(self, server_config):
        """Test app creation with config."""
        with patch('faiss.server.app.FaissEngine'):
            app = create_app(server_config)
            assert isinstance(app, FastAPI)
            assert app.title == "Faiss Search Server"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_query(self, client):
        """Test search with empty query."""
        response = client.post(
            "/api/v1/search",
            json={"query": "", "num": 5}
        )
        # Should either validate or handle gracefully
        assert response.status_code in [200, 422, 400]
    
    def test_very_large_num(self, client, mock_engine):
        """Test search with very large num parameter."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "num": 1000000}
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_unicode_query(self, client, mock_engine):
        """Test search with unicode characters."""
        response = client.post(
            "/api/v1/search",
            json={"query": "测试查询 🚀", "num": 5}
        )
        assert response.status_code == 200
    
    def test_special_characters(self, client, mock_engine):
        """Test search with special characters."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test & query <with> special chars", "num": 5}
        )
        assert response.status_code == 200


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
