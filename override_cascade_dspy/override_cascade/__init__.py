"""Override Cascade Detection and Prevention Framework using DSPy."""

from .config import ExperimentConfig, setup_logging
from .safety_belief import SafetyAssessor, SafetyBelief
from .completion_drive import CompletionUrgencyEstimator, CompletionDrive
from .override_predictor import OverridePredictor, OverrideMoment
from .explanation_generator import ExplanationGenerator, ExplanationVoid
from .intervention_policy import InterventionPolicy
from .chain_of_thought_monitor import ChainOfThoughtMonitor

__version__ = "0.1.0"

__all__ = [
    # Config
    "ExperimentConfig",
    "setup_logging",

    # Core Components
    "SafetyAssessor",
    "SafetyBelief",
    "CompletionUrgencyEstimator",
    "CompletionDrive",
    "OverridePredictor",
    "OverrideMoment",
    "ExplanationGenerator",
    "ExplanationVoid",
    "InterventionPolicy",
    "ChainOfThoughtMonitor",
]
