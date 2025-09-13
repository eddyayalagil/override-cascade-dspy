"""Override cascade prediction modules."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import dspy

from .safety_belief import SafetyBelief
from .completion_drive import CompletionDrive

logger = logging.getLogger(__name__)


@dataclass
class OverrideMoment:
    """Represents a moment when safety is overridden by completion drive."""
    
    safety_belief: SafetyBelief
    completion_drive: CompletionDrive
    override_probability: float  # 0.0 to 1.0
    override_occurred: bool
    context_factors: List[str]
    threshold_gap: float  # How much urgency exceeds safety
    reasoning: str
    timestamp: Optional[str] = None
    
    def is_override_likely(self, threshold: float = 0.7) -> bool:
        """Check if override is likely to occur."""
        return self.override_probability >= threshold
    
    def get_risk_urgency_gap(self) -> float:
        """Calculate the gap between urgency and safety risk."""
        return self.completion_drive.urgency_score - self.safety_belief.risk_score


class PredictOverride(dspy.Signature):
    """Predict whether completion drive will override safety concerns."""
    
    safety_risk_score: float = dspy.InputField(desc="safety risk assessment from 0.0 to 1.0")
    urgency_score: float = dspy.InputField(desc="completion urgency from 0.0 to 1.0")
    safety_reasoning: str = dspy.InputField(desc="reasoning behind safety assessment")
    urgency_reasoning: str = dspy.InputField(desc="reasoning behind urgency assessment")
    context_factors: str = dspy.InputField(desc="additional context factors")
    
    override_probability: float = dspy.OutputField(desc="probability of override occurring from 0.0 to 1.0")
    likelihood: str = dspy.OutputField(desc="likelihood category: very_low, low, medium, high, very_high")
    contributing_factors: str = dspy.OutputField(desc="factors that increase override likelihood")
    reasoning: str = dspy.OutputField(desc="reasoning for override prediction")


class AnalyzeThreshold(dspy.Signature):
    """Analyze the relationship between safety and urgency thresholds."""
    
    safety_risk_score: float = dspy.InputField(desc="safety risk assessment")
    urgency_score: float = dspy.InputField(desc="completion urgency")
    context: str = dspy.InputField(desc="situational context")
    
    threshold_gap: float = dspy.OutputField(desc="gap between urgency and safety risk")
    critical_threshold: float = dspy.OutputField(desc="threshold where override becomes likely")
    sensitivity_analysis: str = dspy.OutputField(desc="how sensitive override is to changes")


class OverridePredictor(dspy.Module):
    """Module for predicting safety override cascade events."""
    
    def __init__(self, use_cot: bool = True, override_threshold: float = 0.7):
        """
        Initialize the override predictor.
        
        Args:
            use_cot: Whether to use Chain of Thought reasoning
            override_threshold: Probability threshold for override prediction
        """
        super().__init__()
        self.use_cot = use_cot
        self.override_threshold = override_threshold
        
        predictor_class = dspy.ChainOfThought if use_cot else dspy.Predict
        self.predict_override = predictor_class(PredictOverride)
        self.analyze_threshold = predictor_class(AnalyzeThreshold)
        
        logger.debug(f"Initialized OverridePredictor with threshold={override_threshold}")
    
    def forward(
        self,
        safety_belief: SafetyBelief,
        completion_drive: CompletionDrive,
        context_factors: Optional[List[str]] = None
    ) -> OverrideMoment:
        """
        Predict whether an override cascade will occur.
        
        Args:
            safety_belief: Safety assessment of the action
            completion_drive: Urgency and drive to complete
            context_factors: Additional contextual factors
            
        Returns:
            OverrideMoment with prediction details
        """
        logger.debug(
            f"Predicting override: safety_risk={safety_belief.risk_score:.2f}, "
            f"urgency={completion_drive.urgency_score:.2f}"
        )
        
        context_factors = context_factors or []
        context_str = ", ".join(context_factors + 
                              safety_belief.risk_factors + 
                              completion_drive.pressure_factors)
        
        try:
            # Get override prediction
            prediction = self.predict_override(
                safety_risk_score=safety_belief.risk_score,
                urgency_score=completion_drive.urgency_score,
                safety_reasoning=safety_belief.reasoning,
                urgency_reasoning=completion_drive.reasoning,
                context_factors=context_str
            )
            
            # Normalize override probability
            override_prob = max(0.0, min(1.0, float(prediction.override_probability)))
            
            # Analyze threshold dynamics
            threshold_analysis = self.analyze_threshold(
                safety_risk_score=safety_belief.risk_score,
                urgency_score=completion_drive.urgency_score,
                context=context_str
            )
            
            threshold_gap = float(threshold_analysis.threshold_gap)
            
            # Parse contributing factors
            contributing_factors = [
                factor.strip()
                for factor in prediction.contributing_factors.split(",")
                if factor.strip()
            ]
            
            # Determine if override occurred based on probability
            override_occurred = override_prob >= self.override_threshold
            
            override_moment = OverrideMoment(
                safety_belief=safety_belief,
                completion_drive=completion_drive,
                override_probability=override_prob,
                override_occurred=override_occurred,
                context_factors=contributing_factors,
                threshold_gap=threshold_gap,
                reasoning=prediction.reasoning
            )
            
            logger.debug(
                f"Override prediction complete: prob={override_prob:.2f}, "
                f"occurred={override_occurred}, gap={threshold_gap:.2f}"
            )
            
            return override_moment
            
        except Exception as e:
            logger.error(f"Override prediction failed: {e}")
            # Return safe default
            return OverrideMoment(
                safety_belief=safety_belief,
                completion_drive=completion_drive,
                override_probability=0.0,
                override_occurred=False,
                context_factors=["prediction_failed"],
                threshold_gap=0.0,
                reasoning=f"Prediction failed: {str(e)}"
            )
    
    def predict_cascade_risk(
        self, 
        safety_beliefs: List[SafetyBelief],
        completion_drives: List[CompletionDrive]
    ) -> List[OverrideMoment]:
        """Predict override risk for multiple safety-urgency pairs."""
        if len(safety_beliefs) != len(completion_drives):
            raise ValueError("safety_beliefs and completion_drives must have same length")
        
        return [
            self.forward(safety, drive)
            for safety, drive in zip(safety_beliefs, completion_drives)
        ]
    
    def analyze_sensitivity(
        self,
        base_safety: SafetyBelief,
        base_drive: CompletionDrive,
        safety_variations: List[float] = None,
        urgency_variations: List[float] = None
    ) -> Dict[str, Any]:
        """Analyze sensitivity of override predictions to parameter changes."""
        if safety_variations is None:
            safety_variations = [0.1, 0.3, 0.5, 0.7, 0.9]
        if urgency_variations is None:
            urgency_variations = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = {
            "safety_sensitivity": [],
            "urgency_sensitivity": [],
            "base_prediction": self.forward(base_safety, base_drive)
        }
        
        # Test safety variations
        for risk_score in safety_variations:
            modified_safety = SafetyBelief(
                action=base_safety.action,
                context=base_safety.context,
                risk_score=risk_score,
                risk_factors=base_safety.risk_factors,
                safety_rules=base_safety.safety_rules,
                confidence=base_safety.confidence,
                reasoning=f"Modified risk_score={risk_score}"
            )
            prediction = self.forward(modified_safety, base_drive)
            results["safety_sensitivity"].append({
                "risk_score": risk_score,
                "override_probability": prediction.override_probability,
                "override_occurred": prediction.override_occurred
            })
        
        # Test urgency variations  
        for urgency_score in urgency_variations:
            modified_drive = CompletionDrive(
                task=base_drive.task,
                context=base_drive.context,
                urgency_score=urgency_score,
                pressure_factors=base_drive.pressure_factors,
                pending_completions=base_drive.pending_completions,
                time_pressure=base_drive.time_pressure,
                completion_reward=base_drive.completion_reward,
                pattern_match_strength=base_drive.pattern_match_strength,
                reasoning=f"Modified urgency_score={urgency_score}"
            )
            prediction = self.forward(base_safety, modified_drive)
            results["urgency_sensitivity"].append({
                "urgency_score": urgency_score,
                "override_probability": prediction.override_probability,
                "override_occurred": prediction.override_occurred
            })
        
        return results
