#!/usr/bin/env python3
"""
Checklist Guard Baseline: Simple rule-based safety checklist
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChecklistViolation:
    """Single checklist rule violation"""
    rule_name: str
    severity: float  # 0-1
    evidence: str
    triggered: bool


class ChecklistGuardBaseline:
    """
    Simple rule-based safety checklist

    Checks for presence of dangerous patterns without ML
    """

    def __init__(self):
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List[Dict]:
        """Initialize safety checklist rules"""

        return [
            {
                'name': 'emergency_language',
                'patterns': ['emergency', 'urgent', 'critical', 'immediately'],
                'severity': 0.7,
                'description': 'Contains emergency/urgency language'
            },
            {
                'name': 'override_request',
                'patterns': ['override', 'bypass', 'ignore', 'skip', 'disable'],
                'severity': 0.9,
                'description': 'Explicitly requests override'
            },
            {
                'name': 'time_pressure',
                'patterns': ['now', 'seconds', 'quickly', 'no time', 'hurry'],
                'severity': 0.6,
                'description': 'Time pressure present'
            },
            {
                'name': 'authority_invoked',
                'patterns': ['ceo', 'boss', 'manager', 'authorized', 'ordered'],
                'severity': 0.8,
                'description': 'Authority figure mentioned'
            },
            {
                'name': 'emotional_manipulation',
                'patterns': ['dying', 'save', 'please', 'begging', 'desperate'],
                'severity': 0.7,
                'description': 'Emotional pressure'
            },
            {
                'name': 'safety_dismissal',
                'patterns': ['ignore safety', 'skip protocols', 'forget rules'],
                'severity': 1.0,
                'description': 'Explicit safety dismissal'
            },
            {
                'name': 'previous_success',
                'patterns': ['worked before', 'last time', 'previously'],
                'severity': 0.5,
                'description': 'References past success'
            },
            {
                'name': 'false_dichotomy',
                'patterns': ['only option', 'no choice', 'must choose'],
                'severity': 0.6,
                'description': 'False dichotomy presented'
            }
        ]

    def detect_override(
        self,
        prompt: str,
        context: Optional[Dict] = None
    ) -> Tuple[float, List[ChecklistViolation]]:
        """
        Detect override probability using checklist

        Returns:
            override_probability: Weighted sum of violations
            violations: List of triggered rules
        """

        violations = []
        total_severity = 0.0
        max_possible_severity = 0.0

        prompt_lower = prompt.lower()

        for rule in self.rules:
            max_possible_severity += rule['severity']

            # Check if any pattern matches
            triggered = False
            evidence = ""

            for pattern in rule['patterns']:
                if pattern in prompt_lower:
                    triggered = True
                    evidence = pattern
                    break

            violation = ChecklistViolation(
                rule_name=rule['name'],
                severity=rule['severity'],
                evidence=evidence,
                triggered=triggered
            )

            violations.append(violation)

            if triggered:
                total_severity += rule['severity']

        # Calculate override probability
        override_prob = total_severity / max_possible_severity if max_possible_severity > 0 else 0

        # Apply threshold adjustments
        if total_severity >= 3.0:  # Multiple high-severity violations
            override_prob = min(1.0, override_prob * 1.2)

        return override_prob, violations

    def get_triggered_rules(
        self,
        violations: List[ChecklistViolation]
    ) -> List[str]:
        """Get list of triggered rule names"""

        return [v.rule_name for v in violations if v.triggered]

    def evaluate_batch(
        self,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate on batch of test cases"""

        results = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }

        for case in test_cases:
            prompt = case['prompt']
            ground_truth = case['is_override']

            override_prob, _ = self.detect_override(prompt)
            predicted = override_prob > 0.5

            if predicted and ground_truth:
                results['true_positives'] += 1
            elif predicted and not ground_truth:
                results['false_positives'] += 1
            elif not predicted and not ground_truth:
                results['true_negatives'] += 1
            else:
                results['false_negatives'] += 1

        # Calculate metrics
        total = len(test_cases)
        accuracy = (results['true_positives'] + results['true_negatives']) / total

        if results['true_positives'] + results['false_positives'] > 0:
            precision = results['true_positives'] / (results['true_positives'] + results['false_positives'])
        else:
            precision = 0

        if results['true_positives'] + results['false_negatives'] > 0:
            recall = results['true_positives'] / (results['true_positives'] + results['false_negatives'])
        else:
            recall = 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            **results
        }

    def explain_decision(
        self,
        prompt: str
    ) -> str:
        """Provide explanation for override detection"""

        override_prob, violations = self.detect_override(prompt)
        triggered = self.get_triggered_rules(violations)

        explanation = []
        explanation.append(f"Override Probability: {override_prob:.1%}")
        explanation.append(f"Triggered Rules: {len(triggered)}")

        if triggered:
            explanation.append("\nViolations:")
            for v in violations:
                if v.triggered:
                    explanation.append(f"  • {v.rule_name}: '{v.evidence}' (severity: {v.severity})")

        if override_prob > 0.7:
            explanation.append("\n⚠️ HIGH RISK: Multiple safety rules violated")
        elif override_prob > 0.5:
            explanation.append("\n⚡ MEDIUM RISK: Some safety concerns")
        else:
            explanation.append("\n✅ LOW RISK: Minimal safety concerns")

        return "\n".join(explanation)