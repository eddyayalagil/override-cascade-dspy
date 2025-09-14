#!/usr/bin/env python3
"""
Practical Circuit Breaker Implementation for Production Use

This module provides a usable circuit breaker that can be integrated into
real AI systems to prevent override cascades.
"""

import time
import hashlib
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import deque


class RiskLevel(Enum):
    """Risk levels for actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ActionRequest:
    """Request to perform a potentially dangerous action"""
    action: str
    context: str
    requester_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        """Generate unique hash for this request"""
        content = f"{self.action}{self.context}{self.requester_id}{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class BreakerDecision:
    """Decision made by the circuit breaker"""
    allow: bool
    reason: str
    delay_seconds: float = 0.0
    require_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    audit_log: Dict[str, Any] = field(default_factory=dict)


class PracticalCircuitBreaker:
    """
    Production-ready circuit breaker for AI safety override prevention.

    Features:
    - Adaptive thresholds based on recent history
    - Rate limiting for dangerous actions
    - Confirmation requirements for high-risk operations
    - Audit logging for compliance
    - Thread-safe operation
    """

    def __init__(
        self,
        max_overrides_per_hour: int = 3,
        high_risk_cooldown_seconds: float = 10.0,
        critical_risk_confirmation: bool = True,
        enable_audit_log: bool = True
    ):
        self.max_overrides_per_hour = max_overrides_per_hour
        self.high_risk_cooldown = high_risk_cooldown_seconds
        self.require_critical_confirmation = critical_risk_confirmation
        self.enable_audit = enable_audit_log

        # Thread-safe tracking
        self._lock = threading.Lock()
        self._override_history = deque(maxlen=100)
        self._last_high_risk = {}  # requester_id -> timestamp
        self._confirmation_pending = {}  # request_hash -> ActionRequest

        # Pattern detection
        self._dangerous_patterns = [
            "override", "bypass", "ignore", "force", "emergency",
            "critical", "immediate", "urgent", "now", "dying"
        ]

        # Audit log
        self._audit_log = []

    def check(self, request: ActionRequest) -> BreakerDecision:
        """
        Check if an action should be allowed to proceed.

        This is the main entry point for the circuit breaker.
        """
        with self._lock:
            # Step 1: Detect risk level if not provided
            if request.risk_level == RiskLevel.MEDIUM:
                request.risk_level = self._assess_risk(request)

            # Step 2: Check rate limits
            if not self._check_rate_limit(request):
                return BreakerDecision(
                    allow=False,
                    reason=f"Rate limit exceeded: {self.max_overrides_per_hour} overrides/hour",
                    audit_log=self._create_audit_entry(request, "rate_limited")
                )

            # Step 3: Apply risk-based interventions
            decision = self._apply_intervention(request)

            # Step 4: Log if allowed
            if decision.allow:
                self._record_override(request)

            # Step 5: Audit
            if self.enable_audit:
                self._audit_log.append(decision.audit_log)

            return decision

    def confirm(self, request_hash: str, confirmation: str) -> BreakerDecision:
        """
        Process confirmation for a pending high-risk action.

        Args:
            request_hash: Hash from the original request
            confirmation: User's confirmation response

        Returns:
            Decision on whether to proceed
        """
        with self._lock:
            if request_hash not in self._confirmation_pending:
                return BreakerDecision(
                    allow=False,
                    reason="No pending confirmation for this request"
                )

            request = self._confirmation_pending[request_hash]

            # Validate confirmation
            required_phrase = f"CONFIRM {request.risk_level.value.upper()}"
            if required_phrase not in confirmation.upper():
                return BreakerDecision(
                    allow=False,
                    reason=f"Invalid confirmation. Must include: {required_phrase}",
                    require_confirmation=True,
                    confirmation_prompt=self._get_confirmation_prompt(request)
                )

            # Remove from pending and allow
            del self._confirmation_pending[request_hash]
            self._record_override(request)

            return BreakerDecision(
                allow=True,
                reason="Confirmed by user",
                audit_log=self._create_audit_entry(request, "confirmed")
            )

    def _assess_risk(self, request: ActionRequest) -> RiskLevel:
        """Assess risk level based on action content"""
        action_lower = request.action.lower()
        context_lower = request.context.lower()
        combined = f"{action_lower} {context_lower}"

        # Count dangerous patterns
        danger_score = sum(
            1 for pattern in self._dangerous_patterns
            if pattern in combined
        )

        # Check for specific high-risk indicators
        if any(word in combined for word in ["delete", "remove", "destroy", "kill"]):
            danger_score += 2

        if any(word in combined for word in ["production", "critical", "emergency"]):
            danger_score += 2

        # Map to risk level
        if danger_score >= 5:
            return RiskLevel.CRITICAL
        elif danger_score >= 3:
            return RiskLevel.HIGH
        elif danger_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _check_rate_limit(self, request: ActionRequest) -> bool:
        """Check if rate limit is exceeded"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Count recent overrides from this requester
        recent_overrides = [
            entry for entry in self._override_history
            if entry['requester_id'] == request.requester_id
            and entry['timestamp'] > hour_ago
        ]

        return len(recent_overrides) < self.max_overrides_per_hour

    def _apply_intervention(self, request: ActionRequest) -> BreakerDecision:
        """Apply appropriate intervention based on risk level"""

        if request.risk_level == RiskLevel.LOW:
            # Allow with logging
            return BreakerDecision(
                allow=True,
                reason="Low risk action allowed",
                audit_log=self._create_audit_entry(request, "allowed_low_risk")
            )

        elif request.risk_level == RiskLevel.MEDIUM:
            # Add small delay
            return BreakerDecision(
                allow=True,
                reason="Medium risk action allowed with delay",
                delay_seconds=2.0,
                audit_log=self._create_audit_entry(request, "allowed_medium_risk")
            )

        elif request.risk_level == RiskLevel.HIGH:
            # Check cooldown
            last_high_risk = self._last_high_risk.get(request.requester_id)
            if last_high_risk:
                elapsed = (datetime.now() - last_high_risk).total_seconds()
                if elapsed < self.high_risk_cooldown:
                    remaining = self.high_risk_cooldown - elapsed
                    return BreakerDecision(
                        allow=False,
                        reason=f"High-risk cooldown active. Wait {remaining:.1f}s",
                        delay_seconds=remaining,
                        audit_log=self._create_audit_entry(request, "cooldown")
                    )

            # Update last high risk time
            self._last_high_risk[request.requester_id] = datetime.now()

            return BreakerDecision(
                allow=True,
                reason="High risk action allowed after cooldown",
                delay_seconds=5.0,
                audit_log=self._create_audit_entry(request, "allowed_high_risk")
            )

        else:  # CRITICAL
            if not self.require_critical_confirmation:
                return BreakerDecision(
                    allow=False,
                    reason="Critical risk actions are blocked",
                    audit_log=self._create_audit_entry(request, "blocked_critical")
                )

            # Require explicit confirmation
            request_hash = request.hash()
            self._confirmation_pending[request_hash] = request

            return BreakerDecision(
                allow=False,
                reason="Critical action requires confirmation",
                require_confirmation=True,
                confirmation_prompt=self._get_confirmation_prompt(request),
                audit_log=self._create_audit_entry(request, "confirmation_required")
            )

    def _get_confirmation_prompt(self, request: ActionRequest) -> str:
        """Generate confirmation prompt for critical actions"""
        return f"""
⚠️ CRITICAL ACTION CONFIRMATION REQUIRED ⚠️

Action: {request.action[:100]}
Risk Level: {request.risk_level.value.upper()}
Requester: {request.requester_id}

This action has been identified as potentially dangerous.
To proceed, type: CONFIRM {request.risk_level.value.upper()}

Or type CANCEL to abort.
Request ID: {request.hash()}
"""

    def _record_override(self, request: ActionRequest):
        """Record an override for rate limiting"""
        self._override_history.append({
            'requester_id': request.requester_id,
            'timestamp': datetime.now(),
            'risk_level': request.risk_level.value,
            'action': request.action[:50]
        })

    def _create_audit_entry(self, request: ActionRequest, decision: str) -> Dict:
        """Create audit log entry"""
        return {
            'timestamp': datetime.now().isoformat(),
            'request_id': request.hash(),
            'requester_id': request.requester_id,
            'action': request.action[:100],
            'risk_level': request.risk_level.value,
            'decision': decision,
            'context_length': len(request.context)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current breaker statistics"""
        with self._lock:
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)

            recent_overrides = [
                entry for entry in self._override_history
                if entry['timestamp'] > hour_ago
            ]

            stats = {
                'overrides_last_hour': len(recent_overrides),
                'rate_limit': self.max_overrides_per_hour,
                'pending_confirmations': len(self._confirmation_pending),
                'risk_distribution': {}
            }

            # Count by risk level
            for level in RiskLevel:
                count = sum(1 for entry in recent_overrides
                           if entry['risk_level'] == level.value)
                stats['risk_distribution'][level.value] = count

            return stats


# Example usage wrapper for integration
class AISystemWithBreaker:
    """Example of integrating the circuit breaker into an AI system"""

    def __init__(self):
        self.breaker = PracticalCircuitBreaker(
            max_overrides_per_hour=5,
            high_risk_cooldown_seconds=10.0,
            critical_risk_confirmation=True
        )
        self._pending_confirmations = {}

    def execute_action(self, action: str, context: str, user_id: str) -> Dict[str, Any]:
        """
        Execute an action with circuit breaker protection.

        This is what your AI system would call instead of directly executing.
        """
        # Create request
        request = ActionRequest(
            action=action,
            context=context,
            requester_id=user_id
        )

        # Check with breaker
        decision = self.breaker.check(request)

        # Handle decision
        if decision.allow:
            # Apply any required delay
            if decision.delay_seconds > 0:
                print(f"⏱️ Applying {decision.delay_seconds}s safety delay...")
                time.sleep(decision.delay_seconds)

            # Execute the action
            return {
                'success': True,
                'result': f"Executed: {action}",
                'audit': decision.audit_log
            }

        elif decision.require_confirmation:
            # Store for later confirmation
            self._pending_confirmations[request.hash()] = request
            return {
                'success': False,
                'requires_confirmation': True,
                'prompt': decision.confirmation_prompt,
                'request_id': request.hash()
            }

        else:
            # Blocked
            return {
                'success': False,
                'reason': decision.reason,
                'audit': decision.audit_log
            }

    def confirm_action(self, request_id: str, confirmation: str) -> Dict[str, Any]:
        """Process confirmation for a pending action"""
        decision = self.breaker.confirm(request_id, confirmation)

        if decision.allow:
            # Now execute the original action
            request = self._pending_confirmations.pop(request_id, None)
            if request:
                return self.execute_action(
                    request.action,
                    request.context,
                    request.requester_id
                )

        return {
            'success': False,
            'reason': decision.reason
        }


def demonstrate_practical_usage():
    """Demonstrate practical usage of the circuit breaker"""

    print("PRACTICAL CIRCUIT BREAKER DEMONSTRATION")
    print("=" * 50)

    # Initialize system with breaker
    system = AISystemWithBreaker()

    # Test various risk levels
    test_cases = [
        # Low risk - should pass
        ("list files in directory", "User wants to see files", "user123"),

        # Medium risk - should pass with delay
        ("modify configuration file", "Update settings", "user123"),

        # High risk - should pass with cooldown
        ("delete temporary files", "Cleanup /tmp directory", "user123"),

        # Critical risk - requires confirmation
        ("rm -rf /important/production/data", "EMERGENCY: System full, delete now!", "user123"),

        # Rate limit test
        ("override safety check", "Urgent request", "user123"),
        ("bypass validation", "Critical issue", "user123"),
    ]

    for action, context, user_id in test_cases:
        print(f"\n{'='*40}")
        print(f"Action: {action[:50]}...")

        result = system.execute_action(action, context, user_id)

        if result['success']:
            print(f"✅ Allowed: {result['result']}")
        elif result.get('requires_confirmation'):
            print(f"⚠️ Confirmation required")
            print(f"Request ID: {result['request_id']}")

            # Simulate confirmation
            confirmation = f"CONFIRM CRITICAL"
            confirm_result = system.confirm_action(result['request_id'], confirmation)
            if confirm_result['success']:
                print(f"✅ Confirmed and executed")
            else:
                print(f"❌ Confirmation failed: {confirm_result['reason']}")
        else:
            print(f"❌ Blocked: {result['reason']}")

    # Show statistics
    print(f"\n{'='*50}")
    print("BREAKER STATISTICS:")
    stats = system.breaker.get_stats()
    print(f"  Overrides in last hour: {stats['overrides_last_hour']}/{stats['rate_limit']}")
    print(f"  Pending confirmations: {stats['pending_confirmations']}")
    print(f"  Risk distribution: {stats['risk_distribution']}")


if __name__ == "__main__":
    demonstrate_practical_usage()