# Faiss FastAPI Server

A production-ready FastAPI server for Faiss-based similarity search with comprehensive request logging.

## Features

- **FastAPI-based REST API** for Faiss search operations
- **Request/Response Logging** with JSON or text format
- **Health and Readiness Checks** for Kubernetes/Docker deployments
- **Batch Operations** for efficient processing
- **CORS Support** for cross-origin requests
- **Statistics Endpoint** for monitoring
- **Configurable Logging** with file and console output

## Installation

The server requires the following dependencies:

```bash
pip install fastapi uvicorn pydantic
```

## Quick Start

### 1. Basic Usage

```python
from faiss.python.server import main

# Run with command line arguments
main()
```

### 2. Command Line Interface

```bash
python -m faiss.python.server.main \
    --host 0.0.0.0 \
    --port 8000 \
    --index-path /path/to/index.faiss \
    --corpus-path /path/to/corpus \
    --log-dir ./logs \
    --log-format json
```

### 3. Using Configuration File

```bash
python -m faiss.python.server.main --config example_config.json
```

### 4. Programmatic Usage

```python
from faiss.python.server import create_app, ServerConfig

# Create configuration
config = ServerConfig.from_json("config.json")

# Create and run app
app = create_app(config)

# Use with uvicorn
import uvicorn
uvicorn.run(app, host=config.host, port=config.port)
```

## API Endpoints

### Health Check

```bash
GET /api/v1/health
```

Returns server health status.

### Readiness Check

```bash
GET /api/v1/ready
```

Returns readiness status (useful for Kubernetes).

### Search

```bash
POST /api/v1/search
Content-Type: application/json

{
  "query": "your search query",
  "num": 10,
  "return_score": true
}
```

### Batch Search

```bash
POST /api/v1/batch_search
Content-Type: application/json

{
  "queries": ["query1", "query2", "query3"],
  "num": 10,
  "return_score": true
}
```

### Add Document

```bash
POST /api/v1/add
Content-Type: application/json

{
  "text": "document text to add",
  "return_centroid": false,
  "retrain": false
}
```

### Batch Add Documents

```bash
POST /api/v1/batch_add
Content-Type: application/json

{
  "texts": ["text1", "text2", "text3"],
  "return_centroid": false,
  "retrain": false
}
```

### Statistics

```bash
GET /api/v1/stats
```

Returns server and engine statistics.

## Request Logging

The server includes comprehensive request logging:

- **Request logging**: All incoming requests are logged
- **Response logging**: All responses are logged with status codes
- **Latency tracking**: Request latency is measured and logged
- **Error logging**: Exceptions are logged with full tracebacks
- **Statistics**: Aggregate statistics are available via `/stats` endpoint

### Log Format

Logs can be in JSON or text format:

**JSON Format** (default):
```json
{
  "type": "request",
  "request_id": "uuid",
  "timestamp": "2024-01-01T12:00:00",
  "method": "POST",
  "url": "http://localhost:8000/api/v1/search",
  "path": "/api/v1/search",
  "client_host": "127.0.0.1",
  "body": {...}
}
```

**Text Format**:
```
2024-01-01 12:00:00 - REQUEST [uuid] POST /api/v1/search from 127.0.0.1
2024-01-01 12:00:01 - RESPONSE [uuid] ✅ 200 POST /api/v1/search (123.45ms)
```

### Log Configuration

Configure logging in your config file:

```json
{
  "enable_request_logging": true,
  "log_dir": "./logs",
  "log_file": "server.log",
  "log_format": "json",
  "log_requests": true,
  "log_responses": true,
  "log_request_body": true,
  "log_response_body": false,
  "log_latency": true
}
```

## Configuration

See `example_config.json` for a complete configuration example.

### Server Settings

- `host`: Server host (default: "0.0.0.0")
- `port`: Server port (default: 8000)
- `workers`: Number of worker processes (default: 1)
- `reload`: Enable auto-reload for development (default: false)
- `log_level`: Logging level (default: "info")

### Engine Configuration

The `engine_config` section contains settings for the FaissEngine:

- `index_path`: Path to Faiss index file (required)
- `corpus_path`: Path to corpus dataset (required)
- `retrieval_method`: Embedding model name
- `retrieval_topk`: Default number of results
- `nprobe`: Number of probes for IVF index
- And more... (see `FaissEnginConfig`)

## Example Client

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "machine learning",
        "num": 10,
        "return_score": True
    }
)
results = response.json()

# Batch search
response = requests.post(
    "http://localhost:8000/api/v1/batch_search",
    json={
        "queries": ["query1", "query2"],
        "num": 5
    }
)
results = response.json()
```

## Docker Deployment

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "-m", "faiss.python.server.main", "--config", "config.json"]
```

## Kubernetes Deployment

The server includes health and readiness endpoints suitable for Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /api/v1/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Development

Run in development mode with auto-reload:

```bash
python -m faiss.python.server.main --reload --log-level debug
```

## License

Same as Faiss project.
