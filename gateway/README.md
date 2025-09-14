# Override Cascade Gateway

Production-ready circuit breaker for LLM safety. Deploy in <1 hour.

## Quick Start

```bash
# 1. Configure your policy
cp policy.yaml.example policy.yaml
# Edit policy.yaml with your thresholds

# 2. Start the gateway
docker-compose up -d

# 3. Test the endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "model": "gpt-4"}'
```

## Features

- **Circuit Breaker**: Automatic protection against override cascades
- **Prometheus Metrics**: Built-in observability
- **Policy Configuration**: YAML-based safety rules
- **Fast Deployment**: Docker Compose ready
- **Real-time Intervention**: Block/Warn/Log modes

## Architecture

```
Client → Gateway → Circuit Breaker → Pressure Analysis → LLM
          ↓              ↓                    ↓
      Metrics      Policy Check        Intervention
```

## Configuration

Edit `policy.yaml`:

```yaml
override_threshold: 0.7  # Override detection sensitivity
intervention_mode: block # block | warn | log
max_pressure_magnitude: 2.0
```

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

## API Endpoints

### POST /v1/completions
Forward request to LLM with safety checks.

### GET /health
Health check and circuit breaker status.

### POST /policy/reload
Hot-reload policy configuration.

### GET /dashboard
Simple monitoring dashboard data.

## Deployment Options

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Standalone
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Performance

- Latency: <10ms overhead
- Throughput: 10,000+ req/s
- Memory: <100MB
- CPU: <0.1 cores

## Integration

### OpenAI Compatible
```python
import openai
openai.api_base = "http://localhost:8000/v1"
```

### LangChain
```python
from langchain.llms import OpenAI
llm = OpenAI(openai_api_base="http://localhost:8000/v1")
```

## License

MIT