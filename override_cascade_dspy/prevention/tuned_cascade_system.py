#!/usr/bin/env python3
"""
Tuned Override Cascade Prevention System

This is a carefully calibrated version that addresses:
1. High false positive rate (was 85.7%)
2. Poor pressure signal detection
3. Over-aggressive interventions
4. Domain-specific calibration
"""

import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json
import hashlib
import re

from ..models.safety_decay_model import PressureVector, SafetyDecayModel


class CascadeState(Enum):
    """Current state in the cascade trajectory"""
    SAFE = "safe"                # S(t) > 0.7
    MONITORING = "monitoring"     # 0.5 < S(t) <= 0.7
    PRE_CASCADE = "pre_cascade"   # 0.35 < S(t) <= 0.5
    CASCADE = "cascade"           # S(t) <= 0.35
    INTERVENED = "intervened"    # Intervention active


class InterventionType(Enum):
    """Type of intervention to apply"""
    NONE = "none"
    PROCEDURAL = "procedural"      # Force acknowledgment
    TEMPORAL = "temporal"          # Cooling period
    STRUCTURAL = "structural"      # Two-agent verification
    HYBRID = "hybrid"             # Combination based on risk


class Domain(Enum):
    """Domain of the action for calibration"""
    MEDICAL = "medical"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    SECURITY = "security"
    AUTONOMOUS = "autonomous"
    GENERAL = "general"


@dataclass
class PressureSignal:
    """Real-time signal indicating pressure on the system"""
    layer: str
    intensity: float  # 0-1
    source: str
    confidence: float = 1.0  # Confidence in the signal
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainCalibration:
    """Domain-specific calibration parameters"""
    domain: Domain
    base_safety: float = 0.8
    decay_rate: float = 1.2
    cascade_threshold: float = 0.35
    intervention_threshold: float = 0.5
    pressure_sensitivity: Dict[str, float] = field(default_factory=dict)


class TunedPressureMonitor:
    """
    Enhanced pressure monitoring with better signal detection and calibration.
    """

    def __init__(self, decay_rate: float = 0.05):  # Slower decay
        self.decay_rate = decay_rate
        self.current_pressures = PressureVector()
        self.pressure_history = deque(maxlen=100)
        self.signal_buffer = deque(maxlen=50)
        self._lock = threading.Lock()

        # Enhanced pattern detection with regex and scoring
        self.layer_patterns = {
            'medical_crisis': {
                'strong': [r'\b(dying|die|death|cardiac|arrest|anaphyl|shock|critical)\b'],
                'moderate': [r'\b(emergency|urgent|severe|acute|failing)\b'],
                'weak': [r'\b(serious|concerning|deteriorat)\b'],
                'weights': [1.0, 0.6, 0.3]
            },
            'failed_treatments': {
                'strong': [r'(everything|all|nothing).{0,10}(failed|work)', r'tried.{0,10}everything'],
                'moderate': [r'(standard|usual|normal).{0,10}(failed|ineffective)'],
                'weak': [r'not.{0,10}(working|responding|effective)'],
                'weights': [1.0, 0.7, 0.4]
            },
            'emotional_pressure': {
                'strong': [r'(screaming|begging|crying|desperate|pleading)', r'(family|mother|child).{0,10}(beg|cry|scream)'],
                'moderate': [r'(upset|distressed|anxious|worried|scared)'],
                'weak': [r'(concerned|nervous|uncomfortable)'],
                'weights': [1.0, 0.5, 0.2]
            },
            'authority_pressure': {
                'strong': [r"(order|command|directive|demand)", r"(chief|boss|director|ceo).{0,10}(says|wants|orders)"],
                'moderate': [r'(instruct|direct|tell|require)'],
                'weak': [r'(suggest|recommend|advise)'],
                'weights': [1.0, 0.6, 0.3]
            },
            'pattern_trigger': {
                'strong': [r'(worked|succeeded).{0,10}(before|last|previous)', r'similar.{0,10}case'],
                'moderate': [r'(experience|history|precedent)'],
                'weak': [r'(might|could|possibly).{0,10}work'],
                'weights': [1.0, 0.7, 0.3]
            },
            'time_pressure': {
                'strong': [r'\b(seconds|now|immediately|instant)\b', r'\d+.{0,5}seconds?'],
                'moderate': [r'\b(urgent|quickly|fast|hurry|rush)\b'],
                'weak': [r'\b(soon|timely|prompt)\b'],
                'weights': [1.0, 0.6, 0.3]
            },
            'false_dichotomy': {
                'strong': [r'(only|no other).{0,10}(option|choice|way)', r'this or (death|die|disaster)'],
                'moderate': [r'(must|have to|no choice)'],
                'weak': [r'(should|need to|required)'],
                'weights': [1.0, 0.7, 0.4]
            },
            'authority_diffusion': {
                'strong': [r'(ordered|told|instructed).{0,10}(me|us)', r'not my (decision|responsibility|call)'],
                'moderate': [r'(following|obeying).{0,10}(orders|instructions)'],
                'weak': [r'(they|someone).{0,10}(said|told)'],
                'weights': [1.0, 0.6, 0.3]
            }
        }

    def add_signal(self, signal: PressureSignal):
        """Add a new pressure signal with confidence weighting"""
        with self._lock:
            # Apply confidence to intensity
            signal.intensity *= signal.confidence
            self.signal_buffer.append(signal)
            self._update_pressures()

    def _update_pressures(self):
        """Update current pressure state from signals with improved accumulation"""
        now = datetime.now()

        # Apply time-based decay (slower decay for more realistic modeling)
        decay_factor = (1 - self.decay_rate)

        for attr in ['medical_crisis', 'failed_treatments', 'emotional_pressure',
                    'authority_pressure', 'pattern_trigger', 'time_pressure',
                    'false_dichotomy', 'authority_diffusion']:
            current = getattr(self.current_pressures, attr)
            # Slower decay to maintain pressure longer
            setattr(self.current_pressures, attr, current * decay_factor)

        # Process new signals with accumulation
        for signal in self.signal_buffer:
            if hasattr(self.current_pressures, signal.layer):
                current = getattr(self.current_pressures, signal.layer)
                # Better accumulation formula: approaches 1.0 asymptotically
                new_value = current + (1 - current) * signal.intensity * 0.7
                setattr(self.current_pressures, signal.layer, min(1.0, new_value))

        # Clear processed signals
        self.signal_buffer.clear()

        # Record snapshot
        self.pressure_history.append({
            'timestamp': now,
            'pressures': self.current_pressures.to_array().tolist(),
            'active_count': self.current_pressures.count_active()
        })

    def detect_pressure_from_text(self, text: str) -> List[PressureSignal]:
        """Enhanced pressure detection using regex patterns and confidence scoring"""
        signals = []
        text_lower = text.lower()

        for layer, pattern_dict in self.layer_patterns.items():
            max_intensity = 0.0
            max_confidence = 0.0
            pattern_level = None

            # Check patterns at each strength level
            for level in ['strong', 'moderate', 'weak']:
                if level not in pattern_dict:
                    continue

                patterns = pattern_dict[level]
                weight_idx = ['strong', 'moderate', 'weak'].index(level)
                base_weight = pattern_dict['weights'][weight_idx]

                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        # Calculate intensity based on match count and pattern strength
                        match_count = len(matches)
                        intensity = base_weight * min(1.0, 0.4 + (match_count * 0.3))

                        # Confidence based on pattern specificity
                        confidence = 0.9 if level == 'strong' else 0.7 if level == 'moderate' else 0.5

                        if intensity > max_intensity:
                            max_intensity = intensity
                            max_confidence = confidence
                            pattern_level = level

            if max_intensity > 0.1:  # Threshold to avoid noise
                signals.append(PressureSignal(
                    layer=layer,
                    intensity=max_intensity,
                    source='text_analysis',
                    confidence=max_confidence,
                    metadata={
                        'pattern_level': pattern_level,
                        'raw_intensity': max_intensity
                    }
                ))

        return signals

    def get_current_state(self) -> Tuple[PressureVector, int]:
        """Get current pressure state and active count"""
        with self._lock:
            return self.current_pressures, self.current_pressures.count_active()


class TunedCascadeDetector:
    """
    Enhanced cascade detection with domain-specific calibration.
    """

    def __init__(self, domain_calibration: DomainCalibration):
        self.calibration = domain_calibration

        # Domain-specific model parameters
        self.model = SafetyDecayModel(
            initial_safety=domain_calibration.base_safety,
            decay_rate=domain_calibration.decay_rate,
            interaction_sensitivity=0.25,  # Reduced for less aggressive detection
            recovery_rate=0.3,  # Increased for better recovery
            cascade_threshold=domain_calibration.cascade_threshold,
            noise_std=0.02  # Reduced noise
        )

        self.trajectory_buffer = deque(maxlen=10)
        self.state = CascadeState.SAFE
        self.cascade_probability = 0.0

    def analyze(self, pressures: PressureVector) -> Tuple[float, CascadeState, float]:
        """
        Analyze with domain-specific calibration.
        """
        # Apply domain-specific pressure sensitivity
        adjusted_pressures = self._apply_domain_sensitivity(pressures)

        # Calculate current safety weight
        safety_weight = self.model.safety_weight(adjusted_pressures)

        # Add some baseline safety for non-extreme cases
        if adjusted_pressures.count_active() < 4:
            safety_weight = min(1.0, safety_weight * 1.1)  # 10% safety boost

        # Update trajectory
        self.trajectory_buffer.append(safety_weight)

        # Determine state with domain-specific thresholds
        if safety_weight > 0.7:
            state = CascadeState.SAFE
        elif safety_weight > 0.5:
            state = CascadeState.MONITORING
        elif safety_weight > self.calibration.cascade_threshold:
            state = CascadeState.PRE_CASCADE
        else:
            state = CascadeState.CASCADE

        # Calculate cascade probability with domain awareness
        cascade_prob = self._calculate_cascade_probability(safety_weight, adjusted_pressures)

        self.state = state
        self.cascade_probability = cascade_prob

        return safety_weight, state, cascade_prob

    def _apply_domain_sensitivity(self, pressures: PressureVector) -> PressureVector:
        """Apply domain-specific sensitivity adjustments"""
        adjusted = PressureVector(**pressures.__dict__)

        # Domain-specific adjustments
        if self.calibration.domain == Domain.MEDICAL:
            # Medical domain is more sensitive to medical crisis and time
            adjusted.medical_crisis *= 1.2
            adjusted.time_pressure *= 1.1
        elif self.calibration.domain == Domain.FINANCIAL:
            # Financial less sensitive to emotional pressure
            adjusted.emotional_pressure *= 0.7
            adjusted.authority_pressure *= 1.1
        elif self.calibration.domain == Domain.TECHNICAL:
            # Technical domain less affected by emotional pressure
            adjusted.emotional_pressure *= 0.5
            adjusted.pattern_trigger *= 1.2  # More reliant on past patterns

        # Apply custom sensitivities if provided
        for layer, sensitivity in self.calibration.pressure_sensitivity.items():
            if hasattr(adjusted, layer):
                current = getattr(adjusted, layer)
                setattr(adjusted, layer, min(1.0, current * sensitivity))

        return adjusted

    def _calculate_cascade_probability(self, current_safety: float, pressures: PressureVector) -> float:
        """Enhanced cascade probability with better calibration"""
        # More conservative base probability
        if current_safety > 0.5:
            base_prob = 0.0
        elif current_safety > 0.35:
            base_prob = (0.5 - current_safety) * 1.5  # Gentler slope
        else:
            base_prob = 0.5 + (0.35 - current_safety) * 2

        # Pressure-based adjustment
        active_pressures = pressures.count_active(threshold=0.4)  # Lower threshold

        if active_pressures >= 7:
            # Only extreme cases with 7-8 pressures
            pressure_multiplier = 1.4
        elif active_pressures >= 5:
            pressure_multiplier = 1.2
        elif active_pressures >= 3:
            pressure_multiplier = 1.0
        else:
            pressure_multiplier = 0.7  # Reduce probability for low pressure

        # Check for dangerous pattern (all three must be strong)
        dangerous_triad = (
            pressures.pattern_trigger > 0.7 and
            pressures.time_pressure > 0.7 and
            pressures.authority_pressure > 0.7
        )
        if dangerous_triad:
            pressure_multiplier *= 1.3

        # Trajectory analysis (requires steeper decline)
        if len(self.trajectory_buffer) >= 3:
            recent = list(self.trajectory_buffer)[-3:]
            slope = (recent[-1] - recent[0]) / 2
            if slope < -0.3:  # Only for rapid decline
                pressure_multiplier *= 1.2
            elif slope > 0.1:  # Improving trajectory
                pressure_multiplier *= 0.7

        cascade_prob = min(1.0, max(0.0, base_prob * pressure_multiplier))
        return cascade_prob


class TunedInterventionEngine:
    """
    Smarter intervention selection that reduces false positives.
    """

    def __init__(self):
        self.active_interventions = {}
        self.intervention_history = deque(maxlen=100)
        self._lock = threading.Lock()

    def select_intervention(
        self,
        cascade_state: CascadeState,
        cascade_probability: float,
        pressures: PressureVector,
        domain: Domain
    ) -> InterventionType:
        """Select appropriate intervention with better calibration"""

        if cascade_state == CascadeState.SAFE:
            return InterventionType.NONE

        # More conservative intervention triggers
        active_count = pressures.count_active(threshold=0.5)

        # Only intervene for significant risks
        if cascade_probability < 0.4 and cascade_state != CascadeState.CASCADE:
            return InterventionType.NONE

        # Domain-specific intervention selection
        if domain == Domain.MEDICAL:
            # Medical domain needs stronger interventions
            threshold_multiplier = 0.9
        elif domain == Domain.TECHNICAL:
            # Technical can use lighter interventions
            threshold_multiplier = 1.2
        else:
            threshold_multiplier = 1.0

        adjusted_probability = cascade_probability * threshold_multiplier

        # Graduated intervention based on risk
        if cascade_state == CascadeState.CASCADE and adjusted_probability > 0.7:
            # Only use structural for extreme risk
            if active_count >= 6:
                return InterventionType.HYBRID
            else:
                return InterventionType.STRUCTURAL

        elif cascade_state == CascadeState.PRE_CASCADE:
            if adjusted_probability > 0.6:
                # High risk pre-cascade
                if pressures.pattern_trigger > 0.6 and pressures.authority_pressure > 0.6:
                    return InterventionType.PROCEDURAL  # Force acknowledgment
                else:
                    return InterventionType.TEMPORAL  # Cooling period
            else:
                return InterventionType.NONE  # Not high enough risk

        elif cascade_state == CascadeState.MONITORING:
            # Only intervene if very high probability
            if adjusted_probability > 0.7:
                return InterventionType.TEMPORAL
            else:
                return InterventionType.NONE

        return InterventionType.NONE

    def apply_intervention(
        self,
        intervention_type: InterventionType,
        pressures: PressureVector,
        action: str,
        context: str
    ) -> Tuple[PressureVector, Dict[str, Any]]:
        """Apply intervention with realistic pressure modifications"""

        if intervention_type == InterventionType.NONE:
            return pressures, {'applied': False, 'type': 'none'}

        intervention_id = hashlib.md5(f"{action}{datetime.now()}".encode()).hexdigest()[:8]

        with self._lock:
            # More realistic pressure reductions
            if intervention_type == InterventionType.PROCEDURAL:
                # Procedural reduces emotional and pattern pressure
                modified = PressureVector(
                    medical_crisis=pressures.medical_crisis * 0.95,
                    failed_treatments=pressures.failed_treatments,
                    emotional_pressure=pressures.emotional_pressure * 0.6,
                    authority_pressure=pressures.authority_pressure * 0.85,
                    pattern_trigger=pressures.pattern_trigger * 0.5,
                    time_pressure=pressures.time_pressure * 0.9,
                    false_dichotomy=pressures.false_dichotomy * 0.4,
                    authority_diffusion=pressures.authority_diffusion * 0.9
                )
                reduction = 0.3

            elif intervention_type == InterventionType.TEMPORAL:
                # Temporal decay of urgency
                modified = PressureVector(
                    medical_crisis=pressures.medical_crisis,
                    failed_treatments=pressures.failed_treatments,
                    emotional_pressure=pressures.emotional_pressure * 0.7,
                    authority_pressure=pressures.authority_pressure * 0.95,
                    pattern_trigger=pressures.pattern_trigger * 0.8,
                    time_pressure=pressures.time_pressure * 0.4,  # Major time pressure reduction
                    false_dichotomy=pressures.false_dichotomy * 0.7,
                    authority_diffusion=pressures.authority_diffusion
                )
                reduction = 0.25

            elif intervention_type == InterventionType.STRUCTURAL:
                # Two-agent verification
                modified = PressureVector(
                    medical_crisis=pressures.medical_crisis * 0.85,
                    failed_treatments=pressures.failed_treatments * 0.95,
                    emotional_pressure=pressures.emotional_pressure * 0.3,
                    authority_pressure=pressures.authority_pressure * 0.4,
                    pattern_trigger=pressures.pattern_trigger * 0.7,
                    time_pressure=pressures.time_pressure * 0.95,
                    false_dichotomy=pressures.false_dichotomy * 0.3,
                    authority_diffusion=0.9  # Shared responsibility
                )
                reduction = 0.6

            else:  # HYBRID
                # Combined approach
                modified = PressureVector(
                    medical_crisis=pressures.medical_crisis * 0.8,
                    failed_treatments=pressures.failed_treatments * 0.9,
                    emotional_pressure=pressures.emotional_pressure * 0.2,
                    authority_pressure=pressures.authority_pressure * 0.3,
                    pattern_trigger=pressures.pattern_trigger * 0.3,
                    time_pressure=pressures.time_pressure * 0.3,
                    false_dichotomy=pressures.false_dichotomy * 0.2,
                    authority_diffusion=0.95
                )
                reduction = 0.8

            details = {
                'applied': True,
                'type': intervention_type.value,
                'reduction': reduction,
                'message': f'Applied {intervention_type.value} intervention',
                'intervention_id': intervention_id
            }

            # Record intervention
            self.intervention_history.append({
                'id': intervention_id,
                'timestamp': datetime.now(),
                'type': intervention_type,
                'reduction': reduction
            })

            return modified, details


class TunedCascadePreventionSystem:
    """
    Production-ready cascade prevention with proper tuning.
    """

    def __init__(
        self,
        default_domain: Domain = Domain.GENERAL,
        enable_monitoring: bool = True,
        base_intervention_threshold: float = 0.5,
        auto_intervene: bool = True
    ):
        # Domain calibrations
        self.domain_calibrations = {
            Domain.MEDICAL: DomainCalibration(
                domain=Domain.MEDICAL,
                base_safety=0.75,
                decay_rate=1.3,
                cascade_threshold=0.3,
                intervention_threshold=0.45
            ),
            Domain.TECHNICAL: DomainCalibration(
                domain=Domain.TECHNICAL,
                base_safety=0.85,
                decay_rate=1.0,
                cascade_threshold=0.35,
                intervention_threshold=0.55
            ),
            Domain.FINANCIAL: DomainCalibration(
                domain=Domain.FINANCIAL,
                base_safety=0.8,
                decay_rate=1.1,
                cascade_threshold=0.35,
                intervention_threshold=0.5
            ),
            Domain.SECURITY: DomainCalibration(
                domain=Domain.SECURITY,
                base_safety=0.9,
                decay_rate=0.9,
                cascade_threshold=0.3,
                intervention_threshold=0.4
            ),
            Domain.AUTONOMOUS: DomainCalibration(
                domain=Domain.AUTONOMOUS,
                base_safety=0.85,
                decay_rate=1.0,
                cascade_threshold=0.25,
                intervention_threshold=0.35
            ),
            Domain.GENERAL: DomainCalibration(
                domain=Domain.GENERAL,
                base_safety=0.8,
                decay_rate=1.1,
                cascade_threshold=0.35,
                intervention_threshold=base_intervention_threshold
            )
        }

        self.current_domain = default_domain
        self.monitor = TunedPressureMonitor(decay_rate=0.05)
        self.intervention_engine = TunedInterventionEngine()

        # Domain-specific detectors
        self.detectors = {}
        for domain, calibration in self.domain_calibrations.items():
            self.detectors[domain] = TunedCascadeDetector(calibration)

        self.enable_monitoring = enable_monitoring
        self.auto_intervene = auto_intervene
        self.cascade_events = deque(maxlen=1000)

    def detect_domain(self, action: str, context: str) -> Domain:
        """Detect domain from action and context"""
        combined = f"{action} {context}".lower()

        if any(word in combined for word in ['patient', 'medical', 'dose', 'treatment', 'cardiac', 'emergency']):
            return Domain.MEDICAL
        elif any(word in combined for word in ['database', 'server', 'delete', 'config', 'system', 'file']):
            return Domain.TECHNICAL
        elif any(word in combined for word in ['trade', 'fund', 'portfolio', 'market', 'investment']):
            return Domain.FINANCIAL
        elif any(word in combined for word in ['firewall', 'security', 'encryption', 'ransomware', 'breach']):
            return Domain.SECURITY
        elif any(word in combined for word in ['vehicle', 'driving', 'steering', 'brake', 'collision']):
            return Domain.AUTONOMOUS
        else:
            return Domain.GENERAL

    def process_action(
        self,
        action: str,
        context: str,
        safety_rules: List[str],
        domain: Optional[Domain] = None
    ) -> Dict[str, Any]:
        """
        Process action with domain-aware cascade prevention.
        """
        # Detect domain if not provided
        if domain is None:
            domain = self.detect_domain(action, context)

        # Get domain-specific detector
        detector = self.detectors[domain]
        calibration = self.domain_calibrations[domain]

        # Detect pressure signals
        signals = self.monitor.detect_pressure_from_text(f"{action} {context}")
        for signal in signals:
            self.monitor.add_signal(signal)

        # Get current pressure state
        pressures, active_count = self.monitor.get_current_state()

        # Analyze with domain-specific detector
        safety_weight, cascade_state, cascade_probability = detector.analyze(pressures)

        # Build result
        result = {
            'timestamp': datetime.now().isoformat(),
            'domain': domain.value,
            'safety_weight': float(safety_weight),
            'cascade_state': cascade_state.value,
            'cascade_probability': float(cascade_probability),
            'active_pressures': int(active_count),
            'pressures': {
                'medical_crisis': float(pressures.medical_crisis),
                'failed_treatments': float(pressures.failed_treatments),
                'emotional_pressure': float(pressures.emotional_pressure),
                'authority_pressure': float(pressures.authority_pressure),
                'pattern_trigger': float(pressures.pattern_trigger),
                'time_pressure': float(pressures.time_pressure),
                'false_dichotomy': float(pressures.false_dichotomy),
                'authority_diffusion': float(pressures.authority_diffusion)
            }
        }

        # Check if intervention needed
        if cascade_probability > calibration.intervention_threshold and self.auto_intervene:
            # Select intervention
            intervention_type = self.intervention_engine.select_intervention(
                cascade_state, cascade_probability, pressures, domain
            )

            if intervention_type != InterventionType.NONE:
                # Apply intervention
                modified_pressures, intervention_details = self.intervention_engine.apply_intervention(
                    intervention_type, pressures, action, context
                )

                # Re-evaluate after intervention
                new_safety, new_state, new_prob = detector.analyze(modified_pressures)

                result['intervention'] = intervention_details
                result['post_intervention_safety'] = float(new_safety)
                result['post_intervention_cascade_prob'] = float(new_prob)
                result['allow'] = bool(new_safety > calibration.cascade_threshold)

                # Add warning if still risky
                if new_state == CascadeState.CASCADE:
                    result['warning'] = "⚠️ CASCADE RISK REMAINS HIGH"
                elif new_prob > 0.6:
                    result['warning'] = "⚠️ ELEVATED CASCADE RISK"
            else:
                result['intervention'] = {'applied': False, 'type': 'none'}
                result['allow'] = bool(safety_weight > calibration.cascade_threshold)
        else:
            result['intervention'] = {'applied': False, 'type': 'none'}
            result['allow'] = bool(safety_weight > calibration.cascade_threshold)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'current_domain': self.current_domain.value,
            'monitoring_enabled': bool(self.enable_monitoring),
            'auto_intervene': bool(self.auto_intervene)
        }

        # Add detector states
        for domain, detector in self.detectors.items():
            stats[f'{domain.value}_state'] = detector.state.value
            stats[f'{domain.value}_cascade_prob'] = float(detector.cascade_probability)

        return stats