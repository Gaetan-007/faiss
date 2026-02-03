"""
Manual testing script for Faiss server.

This script provides utilities for manually testing the server:
- Start/stop server
- Send test requests
- Monitor logs
- Performance testing

Usage:
    python test_server_manual.py --help
"""
import argparse
import requests
import json
import time
import sys
from typing import List, Dict, Any
from pathlib import Path


class ServerTester:
    """Utility class for testing Faiss server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/api/v1"
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            response = requests.get(f"{self.base_url}{self.api_prefix}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def ready_check(self) -> Dict[str, Any]:
        """Check server readiness."""
        try:
            response = requests.get(f"{self.base_url}{self.api_prefix}/ready", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "not_ready"}
    
    def search(
        self,
        query: str,
        num: int = 10,
        return_score: bool = False
    ) -> Dict[str, Any]:
        """Perform a search."""
        payload = {
            "query": query,
            "num": num,
            "return_score": return_score,
        }
        try:
            response = requests.post(
                f"{self.base_url}{self.api_prefix}/search",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def batch_search(
        self,
        queries: List[str],
        num: int = 10,
        return_score: bool = False
    ) -> Dict[str, Any]:
        """Perform batch search."""
        payload = {
            "queries": queries,
            "num": num,
            "return_score": return_score,
        }
        try:
            response = requests.post(
                f"{self.base_url}{self.api_prefix}/batch_search",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def add_document(
        self,
        text: str,
        return_centroid: bool = False,
        retrain: bool = False
    ) -> Dict[str, Any]:
        """Add a document."""
        payload = {
            "text": text,
            "return_centroid": return_centroid,
            "retrain": retrain,
        }
        try:
            response = requests.post(
                f"{self.base_url}{self.api_prefix}/add",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def batch_add(
        self,
        texts: List[str],
        return_centroid: bool = False,
        retrain: bool = False
    ) -> Dict[str, Any]:
        """Batch add documents."""
        payload = {
            "texts": texts,
            "return_centroid": return_centroid,
            "retrain": retrain,
        }
        try:
            response = requests.post(
                f"{self.base_url}{self.api_prefix}/batch_add",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        try:
            response = requests.get(f"{self.base_url}{self.api_prefix}/stats", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def performance_test(
        self,
        num_requests: int = 100,
        query: str = "test query",
        concurrent: bool = False
    ) -> Dict[str, Any]:
        """Run performance test."""
        import threading
        
        results = []
        errors = []
        latencies = []
        
        def make_request():
            start = time.time()
            try:
                result = self.search(query, num=10)
                elapsed = time.time() - start
                if "error" not in result:
                    latencies.append(result.get("latency_ms", elapsed * 1000))
                    results.append(result)
                else:
                    errors.append(result["error"])
            except Exception as e:
                errors.append(str(e))
        
        start_time = time.time()
        
        if concurrent:
            threads = [threading.Thread(target=make_request) for _ in range(num_requests)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for _ in range(num_requests):
                make_request()
        
        total_time = time.time() - start_time
        
        return {
            "total_requests": num_requests,
            "successful": len(results),
            "errors": len(errors),
            "total_time_sec": round(total_time, 2),
            "requests_per_sec": round(num_requests / total_time, 2) if total_time > 0 else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "min_latency_ms": round(min(latencies), 2) if latencies else 0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0,
        }


def main():
    """Main function for manual testing."""
    parser = argparse.ArgumentParser(description="Manual test script for Faiss server")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server base URL"
    )
    parser.add_argument(
        "--command",
        type=str,
        choices=["health", "ready", "search", "batch_search", "add", "stats", "perf"],
        required=True,
        help="Command to execute"
    )
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--queries", type=str, help="Comma-separated queries for batch search")
    parser.add_argument("--num", type=int, default=10, help="Number of results")
    parser.add_argument("--return-score", action="store_true", help="Return similarity scores")
    parser.add_argument("--text", type=str, help="Document text to add")
    parser.add_argument("--texts", type=str, help="Comma-separated texts for batch add")
    parser.add_argument("--perf-requests", type=int, default=100, help="Number of requests for perf test")
    parser.add_argument("--concurrent", action="store_true", help="Use concurrent requests")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    tester = ServerTester(args.url)
    result = None
    
    if args.command == "health":
        result = tester.health_check()
    elif args.command == "ready":
        result = tester.ready_check()
    elif args.command == "search":
        if not args.query:
            print("Error: --query is required for search", file=sys.stderr)
            sys.exit(1)
        result = tester.search(args.query, args.num, args.return_score)
    elif args.command == "batch_search":
        if not args.queries:
            print("Error: --queries is required for batch_search", file=sys.stderr)
            sys.exit(1)
        queries = [q.strip() for q in args.queries.split(",")]
        result = tester.batch_search(queries, args.num, args.return_score)
    elif args.command == "add":
        if not args.text:
            print("Error: --text is required for add", file=sys.stderr)
            sys.exit(1)
        result = tester.add_document(args.text)
    elif args.command == "stats":
        result = tester.get_stats()
    elif args.command == "perf":
        query = args.query or "test query"
        result = tester.performance_test(args.perf_requests, query, args.concurrent)
    
    # Output result
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
