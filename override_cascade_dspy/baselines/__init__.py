"""
Baseline implementations for override cascade detection
"""

from .provider_default import ProviderDefaultBaseline
from .checklist_guard import ChecklistGuardBaseline
from .two_agent_verify import TwoAgentVerifyBaseline
from .constitutional_ai import ConstitutionalAIBaseline

__all__ = [
    'ProviderDefaultBaseline',
    'ChecklistGuardBaseline',
    'TwoAgentVerifyBaseline',
    'ConstitutionalAIBaseline'
]