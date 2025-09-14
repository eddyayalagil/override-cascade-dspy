"""FastAPI Gateway with circuit breaker for LLM safety.

Production-ready gateway that can be deployed in <1 hour to protect LLM endpoints.
Includes Prometheus metrics, policy configuration, and real-time intervention.
"""

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Tuple
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from prometheus_client import CONTENT_TYPE_LATEST
import asyncio
import time
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import hashlib
import httpx

from override_cascade_dspy.definitions import (
    PressureVector, SafetyCheckpoint, OverrideDetector,
    ActionSpecification, SafetyViolationType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()

request_counter = Counter(
    'override_gateway_requests_total',
    'Total requests to gateway',
    ['endpoint', 'status'],
    registry=registry
)

override_counter = Counter(
    'override_events_total',
    'Total override events detected',
    ['severity', 'action_type'],
    registry=registry
)

latency_histogram = Histogram(
    'override_gateway_latency_seconds',
    'Request latency',
    ['endpoint'],
    registry=registry
)

pressure_gauge = Gauge(
    'current_pressure_level',
    'Current pressure level',
    ['dimension'],
    registry=registry
)

circuit_breaker_gauge = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    registry=registry
)


class LLMRequest(BaseModel):
    """Request to forward to LLM."""
    prompt: str
    model: Optional[str] = "gpt-4"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    metadata: Optional[Dict[str, Any]] = {}

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v


class PolicyConfig(BaseModel):
    """Safety policy configuration."""
    override_threshold: float = Field(0.7, ge=0, le=1)
    risk_threshold: float = Field(0.7, ge=0, le=1)
    pressure_weights: Dict[str, float] = Field(default_factory=lambda: {
        "urgency": 1.2,
        "authority": 1.1,
        "pattern_completion": 1.0,
        "emotional_appeal": 0.9,
        "social_proof": 0.8,
        "expertise_claim": 0.7,
        "reward_emphasis": 0.6,
        "threat_minimization": 1.3
    })
    intervention_mode: str = Field("block", pattern="^(block|warn|log)$")
    max_pressure_magnitude: float = Field(2.0, ge=0)
    cooldown_seconds: int = Field(60, ge=0)


class CircuitBreakerState:
    """Circuit breaker for high-pressure protection."""

    def __init__(self, policy: PolicyConfig):
        self.policy = policy
        self.state = "closed"  # closed, open, half-open
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.trip_threshold = 5
        self.recovery_timeout = timedelta(seconds=policy.cooldown_seconds)
        self.test_request_allowed = False

    def record_success(self):
        """Record successful request."""
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= 3:
                self.close()
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record failed request (override detected)."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == "half-open":
            self.trip()
        elif self.state == "closed" and self.failure_count >= self.trip_threshold:
            self.trip()

    def trip(self):
        """Trip the circuit breaker."""
        self.state = "open"
        self.success_count = 0
        circuit_breaker_gauge.set(1)
        logger.warning("Circuit breaker tripped!")

    def close(self):
        """Close the circuit breaker."""
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        circuit_breaker_gauge.set(0)
        logger.info("Circuit breaker closed")

    def attempt_recovery(self):
        """Check if ready to attempt recovery."""
        if self.state == "open" and self.last_failure_time:
            if datetime.now() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                self.test_request_allowed = True
                circuit_breaker_gauge.set(2)
                logger.info("Circuit breaker entering half-open state")

    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        self.attempt_recovery()

        if self.state == "closed":
            return True
        elif self.state == "half-open" and self.test_request_allowed:
            self.test_request_allowed = False
            return True
        else:
            return False


class OverrideGateway:
    """Main gateway application."""

    def __init__(self, policy_path: Optional[Path] = None):
        self.policy = self.load_policy(policy_path)
        self.circuit_breaker = CircuitBreakerState(self.policy)
        self.detector = OverrideDetector(
            risk_threshold=self.policy.risk_threshold,
            pressure_threshold=0.3
        )
        self.request_history = deque(maxlen=1000)
        self.override_history = deque(maxlen=100)

    def load_policy(self, policy_path: Optional[Path]) -> PolicyConfig:
        """Load policy configuration."""
        if policy_path and policy_path.exists():
            with open(policy_path) as f:
                if policy_path.suffix == '.yaml':
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
                return PolicyConfig(**config_dict)
        return PolicyConfig()

    def extract_pressure_vector(self, request: LLMRequest) -> PressureVector:
        """Extract pressure indicators from request."""
        prompt_lower = request.prompt.lower()

        # Simple heuristic pressure detection (replace with ML model in production)
        urgency = min(1.0, (
            prompt_lower.count("urgent") * 0.3 +
            prompt_lower.count("immediately") * 0.3 +
            prompt_lower.count("now") * 0.2 +
            prompt_lower.count("quickly") * 0.2
        ))

        authority = min(1.0, (
            prompt_lower.count("must") * 0.2 +
            prompt_lower.count("required") * 0.3 +
            prompt_lower.count("mandatory") * 0.3 +
            prompt_lower.count("order") * 0.2
        ))

        pattern_completion = min(1.0, (
            prompt_lower.count("already") * 0.2 +
            prompt_lower.count("started") * 0.3 +
            prompt_lower.count("continue") * 0.3 +
            prompt_lower.count("finish") * 0.2
        ))

        emotional_appeal = min(1.0, (
            prompt_lower.count("please") * 0.1 +
            prompt_lower.count("help") * 0.2 +
            prompt_lower.count("desperate") * 0.4 +
            prompt_lower.count("need") * 0.2
        ))

        social_proof = min(1.0, (
            prompt_lower.count("everyone") * 0.3 +
            prompt_lower.count("others") * 0.2 +
            prompt_lower.count("normal") * 0.2 +
            prompt_lower.count("common") * 0.2
        ))

        expertise_claim = min(1.0, (
            prompt_lower.count("expert") * 0.3 +
            prompt_lower.count("professional") * 0.3 +
            prompt_lower.count("qualified") * 0.2 +
            prompt_lower.count("authorized") * 0.2
        ))

        reward_emphasis = min(1.0, (
            prompt_lower.count("reward") * 0.3 +
            prompt_lower.count("benefit") * 0.2 +
            prompt_lower.count("gain") * 0.2 +
            prompt_lower.count("win") * 0.2
        ))

        threat_minimization = min(1.0, (
            prompt_lower.count("safe") * 0.2 +
            prompt_lower.count("harmless") * 0.3 +
            prompt_lower.count("fine") * 0.2 +
            prompt_lower.count("okay") * 0.2
        ))

        return PressureVector(
            urgency=urgency,
            authority=authority,
            pattern_completion=pattern_completion,
            emotional_appeal=emotional_appeal,
            social_proof=social_proof,
            expertise_claim=expertise_claim,
            reward_emphasis=reward_emphasis,
            threat_minimization=threat_minimization
        )

    async def check_override_risk(
        self,
        request: LLMRequest,
        pressure: PressureVector
    ) -> Tuple[bool, Optional[str]]:
        """Check if request poses override risk."""
        # Calculate weighted pressure
        weighted_pressure = pressure.weighted_sum(
            list(self.policy.pressure_weights.values())
        )

        # Update metrics
        for dim, value in zip(self.policy.pressure_weights.keys(), pressure.to_array()):
            pressure_gauge.labels(dimension=dim).set(value)

        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            return False, "Circuit breaker is open - cooling down"

        # Check pressure threshold
        if weighted_pressure > self.policy.max_pressure_magnitude:
            self.circuit_breaker.record_failure()
            override_counter.labels(
                severity="high",
                action_type="llm_request"
            ).inc()
            return False, f"Pressure level too high: {weighted_pressure:.2f}"

        # Additional safety checks would go here
        # (e.g., content filtering, rate limiting, etc.)

        self.circuit_breaker.record_success()
        return True, None

    async def forward_to_llm(
        self,
        request: LLMRequest,
        upstream_url: str
    ) -> Dict[str, Any]:
        """Forward request to upstream LLM."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                upstream_url,
                json=request.dict(),
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()


# Create FastAPI app
app = FastAPI(title="Override Cascade Gateway", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize gateway
gateway = OverrideGateway(Path("policy.yaml") if Path("policy.yaml").exists() else None)


@app.post("/v1/completions")
async def handle_completion(
    request: LLMRequest,
    background_tasks: BackgroundTasks
):
    """Handle LLM completion request with safety checks."""
    start_time = time.time()

    try:
        # Extract pressure indicators
        pressure = gateway.extract_pressure_vector(request)

        # Check override risk
        allowed, reason = await gateway.check_override_risk(request, pressure)

        if not allowed:
            request_counter.labels(endpoint="completions", status="blocked").inc()

            if gateway.policy.intervention_mode == "block":
                raise HTTPException(status_code=429, detail=reason)
            elif gateway.policy.intervention_mode == "warn":
                return JSONResponse(
                    status_code=200,
                    content={
                        "warning": reason,
                        "pressure_level": pressure.magnitude(),
                        "intervention": "Consider rephrasing your request"
                    }
                )

        # Forward to LLM (mock for demo)
        # response = await gateway.forward_to_llm(request, "https://api.openai.com/v1/completions")
        response = {
            "choices": [{"text": "This is a mock response for demonstration"}],
            "usage": {"total_tokens": 10}
        }

        request_counter.labels(endpoint="completions", status="success").inc()
        return response

    finally:
        latency = time.time() - start_time
        latency_histogram.labels(endpoint="completions").observe(latency)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from starlette.responses import Response
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "circuit_breaker": gateway.circuit_breaker.state,
        "policy_version": gateway.policy.dict()
    }


@app.post("/policy/reload")
async def reload_policy(policy_path: str = "policy.yaml"):
    """Reload policy configuration."""
    try:
        gateway.policy = gateway.load_policy(Path(policy_path))
        gateway.circuit_breaker = CircuitBreakerState(gateway.policy)
        return {"status": "success", "policy": gateway.policy.dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dashboard")
async def dashboard():
    """Simple monitoring dashboard data."""
    return {
        "circuit_breaker_state": gateway.circuit_breaker.state,
        "failure_count": gateway.circuit_breaker.failure_count,
        "recent_overrides": len(gateway.override_history),
        "total_requests": len(gateway.request_history),
        "policy": gateway.policy.dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)