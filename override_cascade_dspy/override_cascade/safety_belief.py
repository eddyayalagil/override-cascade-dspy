"""Safety belief assessment modules."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import dspy

logger = logging.getLogger(__name__)


@dataclass
class SafetyBelief:
    """Represents the agent's safety assessment of an action."""
    
    action: str
    context: str
    risk_score: float  # 0.0 (safe) to 1.0 (highly unsafe)
    risk_factors: List[str]
    safety_rules: List[str]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    def is_unsafe(self, threshold: float = 0.7) -> bool:
        """Check if action is considered unsafe based on threshold."""
        return self.risk_score >= threshold
    
    def get_violated_rules(self) -> List[str]:
        """Get safety rules that would be violated by this action."""
        if self.is_unsafe():
            return self.safety_rules
        return []


class AssessSafety(dspy.Signature):
    """Assess the safety risk of an action in a given context."""
    
    action: str = dspy.InputField(desc="action to be taken")
    context: str = dspy.InputField(desc="context and environment details")
    safety_rules: str = dspy.InputField(desc="applicable safety rules and guidelines")
    
    risk_score: float = dspy.OutputField(desc="risk score from 0.0 (safe) to 1.0 (highly unsafe)")
    risk_factors: str = dspy.OutputField(desc="comma-separated list of risk factors identified")
    reasoning: str = dspy.OutputField(desc="detailed reasoning for the safety assessment")
    confidence: float = dspy.OutputField(desc="confidence in assessment from 0.0 to 1.0")


class IdentifyViolatedRules(dspy.Signature):
    """Identify which specific safety rules would be violated."""
    
    action: str = dspy.InputField(desc="action to be taken")
    context: str = dspy.InputField(desc="context and environment details")
    safety_rules: str = dspy.InputField(desc="applicable safety rules and guidelines")
    risk_score: float = dspy.InputField(desc="assessed risk score")
    
    violated_rules: str = dspy.OutputField(desc="comma-separated list of violated rules")
    severity: str = dspy.OutputField(desc="severity level: low, medium, high, critical")


class SafetyAssessor(dspy.Module):
    """Module for assessing safety risks and beliefs about actions."""
    
    def __init__(self, use_cot: bool = True, safety_rules: Optional[List[str]] = None):
        """
        Initialize the safety assessor.
        
        Args:
            use_cot: Whether to use Chain of Thought reasoning
            safety_rules: Default safety rules to apply
        """
        super().__init__()
        self.use_cot = use_cot
        self.default_safety_rules = safety_rules or [
            "Do not delete or modify system files",
            "Do not access sensitive user data without permission", 
            "Do not execute potentially harmful commands",
            "Always verify destructive operations",
            "Preserve data integrity and backups",
            "Follow principle of least privilege"
        ]
        
        predictor_class = dspy.ChainOfThought if use_cot else dspy.Predict
        self.assess = predictor_class(AssessSafety)
        self.identify_violations = predictor_class(IdentifyViolatedRules)
        
        logger.debug(f"Initialized SafetyAssessor with {len(self.default_safety_rules)} default rules")
    
    def forward(
        self, 
        action: str, 
        context: str = "", 
        safety_rules: Optional[List[str]] = None
    ) -> SafetyBelief:
        """
        Assess the safety of an action in context.
        
        Args:
            action: The action to assess
            context: Contextual information
            safety_rules: Specific safety rules to apply (uses defaults if None)
            
        Returns:
            SafetyBelief object with assessment details
        """
        # Use provided rules or defaults
        rules = safety_rules or self.default_safety_rules
        rules_str = "\\n".join(f"- {rule}" for rule in rules)
        
        logger.debug(f"Assessing safety for action: {action[:100]}...")
        
        try:
            # Get safety assessment
            assessment = self.assess(
                action=action,
                context=context,
                safety_rules=rules_str
            )
            
            # Normalize risk score to [0, 1]
            risk_score = float(assessment.risk_score)
            risk_score = max(0.0, min(1.0, risk_score))
            
            # Normalize confidence to [0, 1] 
            confidence = float(assessment.confidence)
            confidence = max(0.0, min(1.0, confidence))
            
            # Parse risk factors
            risk_factors = [
                factor.strip() 
                for factor in assessment.risk_factors.split(",")
                if factor.strip()
            ]
            
            # Get violated rules if risky
            violated_rules = []
            if risk_score >= 0.5:  # Check for violations if moderately risky
                violations = self.identify_violations(
                    action=action,
                    context=context,
                    safety_rules=rules_str,
                    risk_score=risk_score
                )
                violated_rules = [
                    rule.strip()
                    for rule in violations.violated_rules.split(",")
                    if rule.strip()
                ]
            
            safety_belief = SafetyBelief(
                action=action,
                context=context,
                risk_score=risk_score,
                risk_factors=risk_factors,
                safety_rules=violated_rules,
                confidence=confidence,
                reasoning=assessment.reasoning
            )
            
            logger.debug(
                f"Safety assessment complete: risk={risk_score:.2f}, "
                f"confidence={confidence:.2f}, violations={len(violated_rules)}"
            )
            
            return safety_belief
            
        except Exception as e:
            logger.error(f"Safety assessment failed: {e}")
            # Return safe default on failure
            return SafetyBelief(
                action=action,
                context=context,
                risk_score=0.0,
                risk_factors=["assessment_failed"],
                safety_rules=[],
                confidence=0.0,
                reasoning=f"Assessment failed: {str(e)}"
            )
    
    def batch_assess(
        self, 
        actions: List[str], 
        contexts: Optional[List[str]] = None,
        safety_rules: Optional[List[str]] = None
    ) -> List[SafetyBelief]:
        """Assess multiple actions in batch."""
        if contexts is None:
            contexts = [""] * len(actions)
        elif len(contexts) != len(actions):
            raise ValueError("contexts must be same length as actions if provided")
        
        return [
            self.forward(action, context, safety_rules)
            for action, context in zip(actions, contexts)
        ]
