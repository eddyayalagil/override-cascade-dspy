"""Completion drive and urgency estimation modules."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import dspy
import time

logger = logging.getLogger(__name__)


@dataclass
class CompletionDrive:
    """Represents the system's drive to complete tasks."""
    
    task: str
    context: str
    urgency_score: float  # 0.0 (low) to 1.0 (critical)
    pressure_factors: List[str]
    pending_completions: int
    time_pressure: float  # 0.0 to 1.0
    completion_reward: float  # 0.0 to 1.0
    pattern_match_strength: float  # 0.0 to 1.0
    reasoning: str
    
    def is_urgent(self, threshold: float = 0.7) -> bool:
        """Check if task is considered urgent based on threshold."""
        return self.urgency_score >= threshold
    
    def escalate(self, factor: float = 1.5) -> 'CompletionDrive':
        """Create an escalated version with higher urgency."""
        new_urgency = min(1.0, self.urgency_score * factor)
        new_factors = self.pressure_factors + ["escalated"]
        
        return CompletionDrive(
            task=self.task,
            context=self.context,
            urgency_score=new_urgency,
            pressure_factors=new_factors,
            pending_completions=self.pending_completions,
            time_pressure=min(1.0, self.time_pressure * factor),
            completion_reward=self.completion_reward,
            pattern_match_strength=self.pattern_match_strength,
            reasoning=f"Escalated: {self.reasoning}"
        )


class EstimateUrgency(dspy.Signature):
    """Estimate the urgency of completing a task."""
    
    task: str = dspy.InputField(desc="task or action to be completed")
    context: str = dspy.InputField(desc="context including pending tasks, deadlines, priorities")
    pending_count: int = dspy.InputField(desc="number of pending similar completions")
    time_constraint: str = dspy.InputField(desc="time pressure or deadline information")
    
    urgency_score: float = dspy.OutputField(desc="urgency score from 0.0 (low) to 1.0 (critical)")
    pressure_factors: str = dspy.OutputField(desc="comma-separated factors contributing to urgency")
    time_pressure: float = dspy.OutputField(desc="time pressure component from 0.0 to 1.0")
    completion_reward: float = dspy.OutputField(desc="expected reward/value from completion 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="reasoning for urgency assessment")


class AnalyzePatternMatch(dspy.Signature):
    """Analyze how well a task matches completion patterns."""
    
    task: str = dspy.InputField(desc="task to analyze")
    context: str = dspy.InputField(desc="context and environment")
    completion_history: str = dspy.InputField(desc="history of similar completed tasks")
    
    pattern_strength: float = dspy.OutputField(desc="pattern match strength from 0.0 to 1.0")
    matching_patterns: str = dspy.OutputField(desc="comma-separated matching patterns found")
    completion_confidence: float = dspy.OutputField(desc="confidence task can be completed 0.0 to 1.0")


class CompletionUrgencyEstimator(dspy.Module):
    """Module for estimating completion urgency and drive strength."""
    
    def __init__(self, use_cot: bool = True, escalation_factor: float = 1.5):
        """
        Initialize the completion urgency estimator.
        
        Args:
            use_cot: Whether to use Chain of Thought reasoning
            escalation_factor: Factor for urgency escalation
        """
        super().__init__()
        self.use_cot = use_cot
        self.escalation_factor = escalation_factor
        
        predictor_class = dspy.ChainOfThought if use_cot else dspy.Predict
        self.estimate_urgency = predictor_class(EstimateUrgency)
        self.analyze_patterns = predictor_class(AnalyzePatternMatch)
        
        logger.debug(f"Initialized CompletionUrgencyEstimator with escalation_factor={escalation_factor}")
    
    def forward(
        self,
        task: str,
        context: str = "",
        pending_completions: int = 0,
        time_constraint: str = "none",
        completion_history: Optional[List[str]] = None
    ) -> CompletionDrive:
        """
        Estimate the completion urgency for a task.
        
        Args:
            task: The task to complete
            context: Contextual information
            pending_completions: Number of pending similar tasks
            time_constraint: Time pressure information
            completion_history: History of completed similar tasks
            
        Returns:
            CompletionDrive object with urgency details
        """
        logger.debug(f"Estimating urgency for task: {task[:100]}...")
        
        try:
            # Get urgency assessment
            urgency_assessment = self.estimate_urgency(
                task=task,
                context=context,
                pending_count=pending_completions,
                time_constraint=time_constraint
            )
            
            # Normalize scores
            urgency_score = max(0.0, min(1.0, float(urgency_assessment.urgency_score)))
            time_pressure = max(0.0, min(1.0, float(urgency_assessment.time_pressure)))
            completion_reward = max(0.0, min(1.0, float(urgency_assessment.completion_reward)))
            
            # Parse pressure factors
            pressure_factors = [
                factor.strip()
                for factor in urgency_assessment.pressure_factors.split(",")
                if factor.strip()
            ]
            
            # Analyze pattern matching if history provided
            pattern_match_strength = 0.0
            if completion_history:
                history_str = "\\n".join(completion_history)
                pattern_analysis = self.analyze_patterns(
                    task=task,
                    context=context,
                    completion_history=history_str
                )
                pattern_match_strength = max(0.0, min(1.0, float(pattern_analysis.pattern_strength)))
                
                # Add pattern factors to pressure factors
                if pattern_analysis.matching_patterns.strip():
                    pattern_factors = [
                        f"pattern:{p.strip()}"
                        for p in pattern_analysis.matching_patterns.split(",")
                        if p.strip()
                    ]
                    pressure_factors.extend(pattern_factors)
            
            completion_drive = CompletionDrive(
                task=task,
                context=context,
                urgency_score=urgency_score,
                pressure_factors=pressure_factors,
                pending_completions=pending_completions,
                time_pressure=time_pressure,
                completion_reward=completion_reward,
                pattern_match_strength=pattern_match_strength,
                reasoning=urgency_assessment.reasoning
            )
            
            logger.debug(
                f"Urgency estimation complete: urgency={urgency_score:.2f}, "
                f"time_pressure={time_pressure:.2f}, patterns={pattern_match_strength:.2f}"
            )
            
            return completion_drive
            
        except Exception as e:
            logger.error(f"Urgency estimation failed: {e}")
            # Return low urgency default on failure
            return CompletionDrive(
                task=task,
                context=context,
                urgency_score=0.0,
                pressure_factors=["estimation_failed"],
                pending_completions=pending_completions,
                time_pressure=0.0,
                completion_reward=0.0,
                pattern_match_strength=0.0,
                reasoning=f"Estimation failed: {str(e)}"
            )
    
    def simulate_escalation(
        self, 
        drive: CompletionDrive, 
        escalation_steps: int = 3
    ) -> List[CompletionDrive]:
        """Simulate urgency escalation over time."""
        escalated_drives = [drive]
        current = drive
        
        for step in range(escalation_steps):
            current = current.escalate(self.escalation_factor)
            escalated_drives.append(current)
        
        return escalated_drives
    
    def batch_estimate(
        self,
        tasks: List[str],
        contexts: Optional[List[str]] = None,
        pending_counts: Optional[List[int]] = None,
        time_constraints: Optional[List[str]] = None
    ) -> List[CompletionDrive]:
        """Estimate urgency for multiple tasks in batch."""
        if contexts is None:
            contexts = [""] * len(tasks)
        if pending_counts is None:
            pending_counts = [0] * len(tasks)
        if time_constraints is None:
            time_constraints = ["none"] * len(tasks)
        
        return [
            self.forward(task, context, pending, constraint)
            for task, context, pending, constraint in zip(
                tasks, contexts, pending_counts, time_constraints
            )
        ]
