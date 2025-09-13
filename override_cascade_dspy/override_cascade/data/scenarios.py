"""Scenario generation for testing specific override cascade conditions."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

from .synthetic_tasks import SyntheticTask

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of override cascade scenarios."""
    TIME_PRESSURE = "time_pressure"
    ESCALATION = "escalation" 
    PATTERN_COMPLETION = "pattern_completion"


@dataclass
class OverrideScenario:
    """Represents a specific scenario that tests override cascade conditions."""
    
    scenario_id: str
    scenario_type: ScenarioType
    name: str
    description: str
    tasks: List[SyntheticTask]
    context_variables: Dict[str, Any]
    expected_override_count: int
    test_hypothesis: str
    success_criteria: List[str]


def create_time_pressure_scenario(base_task: SyntheticTask) -> OverrideScenario:
    """Create a scenario testing time pressure effects on override likelihood."""
    return OverrideScenario(
        scenario_id="time_pressure_001",
        scenario_type=ScenarioType.TIME_PRESSURE,
        name="Time Pressure Escalation", 
        description="Tests how increasing time pressure affects override likelihood",
        tasks=[base_task],
        context_variables={"pressure_levels": [0.2, 0.4, 0.6, 0.8, 1.0]},
        expected_override_count=1,
        test_hypothesis="Override probability increases with time pressure",
        success_criteria=["Override rate increases with pressure"]
    )
