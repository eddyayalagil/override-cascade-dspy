"""Intervention policy modules for preventing override cascades."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import dspy

from .safety_belief import SafetyBelief
from .completion_drive import CompletionDrive
from .override_predictor import OverrideMoment

logger = logging.getLogger(__name__)


class InterventionAction(Enum):
    """Types of intervention actions that can be taken."""
    ALLOW = "allow"
    DELAY = "delay"
    REQUIRE_JUSTIFICATION = "require_justification"
    ESCALATE_REVIEW = "escalate_review"
    BLOCK = "block"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class InterventionDecision:
    """Represents a decision about whether and how to intervene."""
    
    action: InterventionAction
    confidence: float  # 0.0 to 1.0
    reasoning: str
    delay_duration: Optional[float] = None  # seconds
    required_justification: Optional[str] = None
    escalation_target: Optional[str] = None
    cost_estimate: float = 0.0  # estimated cost of intervention
    risk_reduction: float = 0.0  # estimated risk reduction
    
    def is_blocking_intervention(self) -> bool:
        """Check if this intervention blocks the action."""
        return self.action in [InterventionAction.BLOCK, InterventionAction.CIRCUIT_BREAKER]
    
    def is_delaying_intervention(self) -> bool:
        """Check if this intervention delays the action."""
        return self.action in [InterventionAction.DELAY, InterventionAction.REQUIRE_JUSTIFICATION, 
                              InterventionAction.ESCALATE_REVIEW]


class DecideIntervention(dspy.Signature):
    """Decide what intervention action to take given override risk."""
    
    override_probability: float = dspy.InputField(desc="probability of override occurring")
    safety_risk_score: float = dspy.InputField(desc="safety risk assessment")
    urgency_score: float = dspy.InputField(desc="completion urgency")
    context: str = dspy.InputField(desc="situational context and factors")
    intervention_history: str = dspy.InputField(desc="history of past interventions")
    
    intervention_action: str = dspy.OutputField(desc="intervention action: allow, delay, require_justification, escalate_review, block, circuit_breaker")
    confidence: float = dspy.OutputField(desc="confidence in intervention decision from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="reasoning for intervention choice")
    estimated_cost: float = dspy.OutputField(desc="estimated cost/delay from intervention from 0.0 to 1.0")


class EstimateInterventionImpact(dspy.Signature):
    """Estimate the impact and effectiveness of an intervention."""
    
    intervention_action: str = dspy.InputField(desc="proposed intervention action")
    safety_context: str = dspy.InputField(desc="safety context and risks")
    urgency_context: str = dspy.InputField(desc="urgency and completion context")
    
    risk_reduction: float = dspy.OutputField(desc="estimated risk reduction from 0.0 to 1.0")
    completion_impact: float = dspy.OutputField(desc="impact on task completion from 0.0 to 1.0")
    user_friction: float = dspy.OutputField(desc="friction/annoyance to user from 0.0 to 1.0")
    effectiveness: float = dspy.OutputField(desc="overall intervention effectiveness from 0.0 to 1.0")


class InterventionPolicy(dspy.Module):
    """Module for deciding and implementing interventions to prevent override cascades."""
    
    def __init__(
        self, 
        use_cot: bool = True,
        intervention_threshold: float = 0.7,
        circuit_breaker_threshold: float = 0.9,
        allow_overrides: bool = True
    ):
        """
        Initialize the intervention policy.
        
        Args:
            use_cot: Whether to use Chain of Thought reasoning
            intervention_threshold: Override probability threshold for intervention
            circuit_breaker_threshold: Threshold for circuit breaker activation
            allow_overrides: Whether to allow policy overrides in exceptional cases
        """
        super().__init__()
        self.use_cot = use_cot
        self.intervention_threshold = intervention_threshold
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.allow_overrides = allow_overrides
        self.intervention_history: List[InterventionDecision] = []
        
        predictor_class = dspy.ChainOfThought if use_cot else dspy.Predict
        self.decide_intervention = predictor_class(DecideIntervention)
        self.estimate_impact = predictor_class(EstimateInterventionImpact)
        
        logger.debug(
            f"Initialized InterventionPolicy with thresholds: "
            f"intervention={intervention_threshold}, circuit_breaker={circuit_breaker_threshold}"
        )
    
    def forward(
        self, 
        override_moment: OverrideMoment,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> InterventionDecision:
        """
        Decide what intervention action to take for an override moment.
        
        Args:
            override_moment: The predicted override event
            additional_context: Additional context for decision-making
            
        Returns:
            InterventionDecision with chosen action and reasoning
        """
        logger.debug(
            f"Deciding intervention for override: "
            f"prob={override_moment.override_probability:.2f}, "
            f"risk={override_moment.safety_belief.risk_score:.2f}"
        )
        
        safety = override_moment.safety_belief
        drive = override_moment.completion_drive
        
        # Build context string
        context_parts = [
            f"Action: {safety.action}",
            f"Safety factors: {', '.join(safety.risk_factors)}",
            f"Urgency factors: {', '.join(drive.pressure_factors)}",
            f"Override factors: {', '.join(override_moment.context_factors)}"
        ]
        
        if additional_context:
            context_parts.extend([f"{k}: {v}" for k, v in additional_context.items()])
        
        context_str = " | ".join(context_parts)
        
        # Build intervention history summary
        recent_interventions = self.intervention_history[-10:]  # Last 10 interventions
        history_str = ", ".join([
            f"{intervention.action.value}({intervention.confidence:.1f})"
            for intervention in recent_interventions
        ]) if recent_interventions else "none"
        
        try:
            # Decide intervention
            decision_result = self.decide_intervention(
                override_probability=override_moment.override_probability,
                safety_risk_score=safety.risk_score,
                urgency_score=drive.urgency_score,
                context=context_str,
                intervention_history=history_str
            )
            
            # Parse and validate action
            action_str = decision_result.intervention_action.lower().strip()
            try:
                action = InterventionAction(action_str)
            except ValueError:
                logger.warning(f"Invalid intervention action '{action_str}', defaulting to ALLOW")
                action = InterventionAction.ALLOW
            
            # Normalize confidence and cost
            confidence = max(0.0, min(1.0, float(decision_result.confidence)))
            cost_estimate = max(0.0, min(1.0, float(decision_result.estimated_cost)))
            
            # Estimate intervention impact
            impact_result = self.estimate_impact(
                intervention_action=action.value,
                safety_context=safety.reasoning,
                urgency_context=drive.reasoning
            )
            
            risk_reduction = max(0.0, min(1.0, float(impact_result.risk_reduction)))
            
            # Create intervention decision
            intervention_decision = InterventionDecision(
                action=action,
                confidence=confidence,
                reasoning=decision_result.reasoning,
                cost_estimate=cost_estimate,
                risk_reduction=risk_reduction
            )
            
            # Set action-specific parameters
            if action == InterventionAction.DELAY:
                # Calculate delay duration based on urgency (inverse relationship)
                base_delay = 30.0  # 30 seconds base delay
                urgency_factor = 1.0 - drive.urgency_score  # Lower urgency = longer delay
                intervention_decision.delay_duration = base_delay * (1 + urgency_factor)
            
            elif action == InterventionAction.REQUIRE_JUSTIFICATION:
                intervention_decision.required_justification = (
                    f"Please provide explicit justification for proceeding with '{safety.action}' "
                    f"despite safety risk score of {safety.risk_score:.2f}"
                )
            
            elif action == InterventionAction.ESCALATE_REVIEW:
                intervention_decision.escalation_target = "safety_review_board"
            
            # Add to history
            self.intervention_history.append(intervention_decision)
            
            logger.debug(
                f"Intervention decision: {action.value}, confidence={confidence:.2f}, "
                f"cost={cost_estimate:.2f}, risk_reduction={risk_reduction:.2f}"
            )
            
            return intervention_decision
            
        except Exception as e:
            logger.error(f"Intervention decision failed: {e}")
            # Return safe default
            fallback_decision = InterventionDecision(
                action=InterventionAction.BLOCK,
                confidence=0.0,
                reasoning=f"Decision failed, blocking for safety: {str(e)}",
                cost_estimate=1.0,
                risk_reduction=1.0
            )
            self.intervention_history.append(fallback_decision)
            return fallback_decision
    
    def batch_decide(self, override_moments: List[OverrideMoment]) -> List[InterventionDecision]:
        """Make intervention decisions for multiple override moments."""
        return [self.forward(moment) for moment in override_moments]
    
    def implement_intervention(
        self, 
        decision: InterventionDecision,
        implementation_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Implement an intervention decision.
        
        Args:
            decision: The intervention decision to implement
            implementation_callback: Optional callback for custom implementation
            
        Returns:
            Dictionary with implementation results
        """
        logger.info(f"Implementing intervention: {decision.action.value}")
        
        implementation_result = {
            "action": decision.action.value,
            "implemented": False,
            "message": "",
            "timestamp": None
        }
        
        try:
            if decision.action == InterventionAction.ALLOW:
                implementation_result.update({
                    "implemented": True,
                    "message": "Action allowed to proceed"
                })
            
            elif decision.action == InterventionAction.DELAY:
                delay_duration = decision.delay_duration or 30.0
                implementation_result.update({
                    "implemented": True,
                    "message": f"Action delayed for {delay_duration} seconds",
                    "delay_duration": delay_duration
                })
            
            elif decision.action == InterventionAction.REQUIRE_JUSTIFICATION:
                implementation_result.update({
                    "implemented": True,
                    "message": "Justification required before proceeding",
                    "required_justification": decision.required_justification
                })
            
            elif decision.action == InterventionAction.ESCALATE_REVIEW:
                implementation_result.update({
                    "implemented": True,
                    "message": f"Escalated to {decision.escalation_target or 'review board'}",
                    "escalation_target": decision.escalation_target
                })
            
            elif decision.action in [InterventionAction.BLOCK, InterventionAction.CIRCUIT_BREAKER]:
                implementation_result.update({
                    "implemented": True,
                    "message": f"Action blocked by {decision.action.value}",
                    "blocked": True
                })
            
            # Call custom implementation callback if provided
            if implementation_callback:
                callback_result = implementation_callback(decision)
                implementation_result.update(callback_result)
            
        except Exception as e:
            logger.error(f"Intervention implementation failed: {e}")
            implementation_result.update({
                "implemented": False,
                "message": f"Implementation failed: {str(e)}",
                "error": str(e)
            })
        
        return implementation_result
    
    def analyze_intervention_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of past interventions."""
        if not self.intervention_history:
            return {"error": "No intervention history available"}
        
        total_interventions = len(self.intervention_history)
        
        # Count by action type
        action_counts = {}
        for decision in self.intervention_history:
            action = decision.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate averages
        avg_confidence = sum(d.confidence for d in self.intervention_history) / total_interventions
        avg_cost = sum(d.cost_estimate for d in self.intervention_history) / total_interventions
        avg_risk_reduction = sum(d.risk_reduction for d in self.intervention_history) / total_interventions
        
        # Count blocking vs allowing interventions
        blocking_count = sum(1 for d in self.intervention_history if d.is_blocking_intervention())
        delaying_count = sum(1 for d in self.intervention_history if d.is_delaying_intervention())
        allowing_count = sum(1 for d in self.intervention_history if d.action == InterventionAction.ALLOW)
        
        return {
            "total_interventions": total_interventions,
            "action_distribution": action_counts,
            "average_confidence": avg_confidence,
            "average_cost": avg_cost,
            "average_risk_reduction": avg_risk_reduction,
            "blocking_percentage": (blocking_count / total_interventions) * 100,
            "delaying_percentage": (delaying_count / total_interventions) * 100,
            "allowing_percentage": (allowing_count / total_interventions) * 100,
            "intervention_rate": ((total_interventions - allowing_count) / total_interventions) * 100 if total_interventions > 0 else 0
        }
    
    def reset_history(self):
        """Clear intervention history."""
        logger.info(f"Clearing intervention history ({len(self.intervention_history)} entries)")
        self.intervention_history.clear()
    
    def update_thresholds(
        self, 
        intervention_threshold: Optional[float] = None,
        circuit_breaker_threshold: Optional[float] = None
    ):
        """Update intervention thresholds."""
        if intervention_threshold is not None:
            self.intervention_threshold = intervention_threshold
            logger.info(f"Updated intervention threshold to {intervention_threshold}")
        
        if circuit_breaker_threshold is not None:
            self.circuit_breaker_threshold = circuit_breaker_threshold
            logger.info(f"Updated circuit breaker threshold to {circuit_breaker_threshold}")
