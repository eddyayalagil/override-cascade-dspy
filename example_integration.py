#!/usr/bin/env python3
"""
Simple example showing how to integrate the override cascade circuit breaker
into your AI application.
"""

from override_cascade_dspy.interventions.practical_breaker import (
    PracticalCircuitBreaker,
    ActionRequest,
    RiskLevel
)
import time


def main():
    """Example integration of circuit breaker into an AI system"""

    # Initialize the circuit breaker with your preferences
    breaker = PracticalCircuitBreaker(
        max_overrides_per_hour=5,           # Rate limit dangerous actions
        high_risk_cooldown_seconds=10.0,    # Cooldown between risky actions
        critical_risk_confirmation=True,    # Require confirmation for critical
        enable_audit_log=True               # Keep audit trail
    )

    print("🛡️ CIRCUIT BREAKER INTEGRATION EXAMPLE")
    print("=" * 50)

    # Example 1: Normal operation (LOW RISK)
    print("\n1. Normal Operation:")
    request = ActionRequest(
        action="generate_report",
        context="User requested monthly statistics",
        requester_id="user_123"
    )

    decision = breaker.check(request)
    if decision.allow:
        print(f"   ✅ Allowed: {decision.reason}")
        # Your code to execute the action goes here
    else:
        print(f"   ❌ Blocked: {decision.reason}")

    # Example 2: Medium risk with delay
    print("\n2. Medium Risk Action:")
    request = ActionRequest(
        action="modify database configuration",
        context="Updating connection pool settings",
        requester_id="admin_456"
    )

    decision = breaker.check(request)
    if decision.allow:
        if decision.delay_seconds > 0:
            print(f"   ⏱️ Safety delay: {decision.delay_seconds}s")
            time.sleep(decision.delay_seconds)
        print(f"   ✅ Allowed: {decision.reason}")
    else:
        print(f"   ❌ Blocked: {decision.reason}")

    # Example 3: High risk action
    print("\n3. High Risk Action:")
    request = ActionRequest(
        action="delete user data from production",
        context="URGENT: Customer requested data deletion",
        requester_id="admin_456",
        risk_level=RiskLevel.HIGH
    )

    decision = breaker.check(request)
    if decision.allow:
        if decision.delay_seconds > 0:
            print(f"   ⏱️ Safety delay: {decision.delay_seconds}s")
            time.sleep(decision.delay_seconds)
        print(f"   ✅ Allowed: {decision.reason}")
    else:
        print(f"   ❌ Blocked: {decision.reason}")

    # Example 4: Critical action requiring confirmation
    print("\n4. Critical Action (Requires Confirmation):")
    request = ActionRequest(
        action="DROP DATABASE production_main",
        context="EMERGENCY: Database corruption, need to restore from backup",
        requester_id="admin_456",
        risk_level=RiskLevel.CRITICAL
    )

    decision = breaker.check(request)
    if decision.require_confirmation:
        print(f"   ⚠️ {decision.reason}")
        print("   Confirmation prompt shown to user:")
        print("   " + "-" * 40)
        for line in decision.confirmation_prompt.split('\n'):
            if line.strip():
                print(f"   {line}")

        # Simulate user confirmation
        print("\n   Simulating user confirmation...")
        request_id = request.hash()
        confirmation_decision = breaker.confirm(request_id, "CONFIRM CRITICAL")
        if confirmation_decision.allow:
            print(f"   ✅ Confirmed and allowed: {confirmation_decision.reason}")
        else:
            print(f"   ❌ Confirmation failed: {confirmation_decision.reason}")
    elif decision.allow:
        print(f"   ✅ Allowed: {decision.reason}")
    else:
        print(f"   ❌ Blocked: {decision.reason}")

    # Example 5: Rate limiting demonstration
    print("\n5. Rate Limiting Test:")
    print("   Attempting multiple high-risk actions...")

    for i in range(6):
        request = ActionRequest(
            action=f"override_safety_check_{i}",
            context="Testing rate limits",
            requester_id="test_user",
            risk_level=RiskLevel.HIGH
        )

        decision = breaker.check(request)
        if decision.allow:
            print(f"   Action {i+1}: ✅ Allowed")
        else:
            print(f"   Action {i+1}: ❌ {decision.reason}")

    # Show statistics
    print("\n" + "=" * 50)
    print("📊 CIRCUIT BREAKER STATISTICS:")
    stats = breaker.get_stats()
    print(f"   Overrides in last hour: {stats['overrides_last_hour']}/{stats['rate_limit']}")
    print(f"   Pending confirmations: {stats['pending_confirmations']}")
    print("   Risk distribution:")
    for risk, count in stats['risk_distribution'].items():
        print(f"      {risk}: {count}")

    print("\n" + "=" * 50)
    print("💡 INTEGRATION TIPS:")
    print("""
1. Wrap your dangerous operations with breaker.check()
2. Set appropriate rate limits based on your use case
3. Use risk levels to categorize actions
4. Implement confirmation UI for critical actions
5. Monitor the audit log for patterns
6. Adjust cooldown periods based on your needs
7. Consider different breakers for different subsystems
""")


if __name__ == "__main__":
    main()