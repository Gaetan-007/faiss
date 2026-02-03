"""
API routes for Faiss search server.
"""
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel, Field

from .app import get_engine, get_logger
from .logger import RequestLogger
from ..engine import BaseEngine


router = APIRouter()


# Request/Response models
class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query string")
    num: Optional[int] = Field(None, description="Number of results to return")
    return_score: bool = Field(False, description="Whether to return similarity scores")


class BatchSearchRequest(BaseModel):
    """Batch search request model."""
    queries: List[str] = Field(..., description="List of search query strings")
    num: Optional[int] = Field(None, description="Number of results per query")
    return_score: bool = Field(False, description="Whether to return similarity scores")


class AddRequest(BaseModel):
    """Add document request model."""
    text: str = Field(..., description="Document text to add")
    return_centroid: bool = Field(False, description="Whether to return assigned centroid")
    retrain: bool = Field(False, description="Whether to retrain index after adding")


class BatchAddRequest(BaseModel):
    """Batch add documents request model."""
    texts: List[str] = Field(..., description="List of document texts to add")
    return_centroid: bool = Field(False, description="Whether to return assigned centroids")
    retrain: bool = Field(False, description="Whether to retrain index after adding")


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    scores: Optional[List[float]] = Field(None, description="Similarity scores (if requested)")
    latency_ms: float = Field(..., description="Request latency in milliseconds")


class BatchSearchResponse(BaseModel):
    """Batch search response model."""
    results: List[List[Dict[str, Any]]] = Field(..., description="Search results for each query")
    scores: Optional[List[List[float]]] = Field(None, description="Similarity scores (if requested)")
    latency_ms: float = Field(..., description="Request latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    engine_ready: bool = Field(..., description="Whether engine is ready")
    timestamp: str = Field(..., description="Current timestamp")


class StatsResponse(BaseModel):
    """Statistics response model."""
    engine_stats: Dict[str, Any] = Field(..., description="Engine statistics")
    logger_stats: Dict[str, Any] = Field(..., description="Logger statistics")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    engine: BaseEngine = Depends(get_engine),
):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/ready")
async def ready_check(
    engine: BaseEngine = Depends(get_engine),
):
    """Readiness check endpoint."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not ready"
        )
    return {"status": "ready"}


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    engine: BaseEngine = Depends(get_engine),
    logger: RequestLogger = Depends(get_logger),
):
    """
    Search for similar documents.
    
    Args:
        request: Search request
        engine: Engine instance
        logger: Logger instance
        
    Returns:
        Search results
    """
    start_time = time.time()
    
    try:
        result = engine.search(
            query=request.query,
            num=request.num,
            return_score=request.return_score,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if request.return_score:
            results, scores = result
            return {
                "results": results,
                "scores": scores,
                "latency_ms": round(latency_ms, 2),
            }
        else:
            return {
                "results": result,
                "scores": None,
                "latency_ms": round(latency_ms, 2),
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/batch_search", response_model=BatchSearchResponse)
async def batch_search(
    request: BatchSearchRequest,
    engine: BaseEngine = Depends(get_engine),
    logger: RequestLogger = Depends(get_logger),
):
    """
    Batch search for similar documents.
    
    Args:
        request: Batch search request
        engine: Engine instance
        logger: Logger instance
        
    Returns:
        Batch search results
    """
    start_time = time.time()
    
    if not request.queries:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="queries must not be empty",
        )

    try:
        result = engine.batch_search(
            query_list=request.queries,
            num=request.num,
            return_score=request.return_score,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        expected_len = len(request.queries)
        if request.return_score:
            results, scores = result
            if results is None:
                results = []
            if scores is None:
                scores = []
            if len(results) < expected_len:
                results = results + [[] for _ in range(expected_len - len(results))]
            elif len(results) > expected_len:
                results = results[:expected_len]
            if len(scores) < expected_len:
                scores = scores + [[] for _ in range(expected_len - len(scores))]
            elif len(scores) > expected_len:
                scores = scores[:expected_len]
            return {
                "results": results,
                "scores": scores,
                "latency_ms": round(latency_ms, 2),
            }
        else:
            results = result if result is not None else []
            if len(results) < expected_len:
                results = results + [[] for _ in range(expected_len - len(results))]
            elif len(results) > expected_len:
                results = results[:expected_len]
            return {
                "results": results,
                "scores": None,
                "latency_ms": round(latency_ms, 2),
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )


@router.post("/add")
async def add_document(
    request: AddRequest,
    engine: BaseEngine = Depends(get_engine),
):
    """
    Add a document to the index.
    
    Args:
        request: Add document request
        engine: Engine instance
        
    Returns:
        Add operation result
    """
    try:
        result = engine._add(
            query=request.text,
            return_centroid=request.return_centroid,
            retrain=request.retrain,
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Add document failed: {str(e)}"
        )


@router.post("/batch_add")
async def batch_add_documents(
    request: BatchAddRequest,
    engine: BaseEngine = Depends(get_engine),
):
    """
    Batch add documents to the index.
    
    Args:
        request: Batch add documents request
        engine: Engine instance
        
    Returns:
        Batch add operation result
    """
    try:
        result = engine._batch_add(
            query_list=request.texts,
            return_centroid=request.return_centroid,
            retrain=request.retrain,
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch add documents failed: {str(e)}"
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    request: Request,
    engine: BaseEngine = Depends(get_engine),
    logger: RequestLogger = Depends(get_logger),
):
    """
    Get server statistics.
    
    Args:
        engine: Engine instance
        logger: Logger instance
        
    Returns:
        Statistics
    """
    engine_stats = {}
    if hasattr(engine, "finished_requests"):
        engine_stats = {
            "finished_requests": engine.finished_requests,
            "request_batch_size": getattr(engine, "request_batch_size", 0),
        }
    
    state_logger = getattr(request.app.state, "logger", None)
    active_logger = state_logger or logger
    if active_logger is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Logger not initialized",
        )
    logger_stats = active_logger.get_stats()
    
    return {
        "engine_stats": engine_stats,
        "logger_stats": logger_stats,
    }


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Faiss Search Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "ready": "/api/v1/ready",
            "search": "/api/v1/search",
            "batch_search": "/api/v1/batch_search",
            "add": "/api/v1/add",
            "batch_add": "/api/v1/batch_add",
            "stats": "/api/v1/stats",
        }
    }
