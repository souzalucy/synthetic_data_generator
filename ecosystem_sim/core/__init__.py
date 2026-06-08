"""Core causal inference engine"""

from .user_agent import (
    IncomeLevel,
    TechSavviness,
    PrivacySensitivity,
    Device,
    Persona,
    User,
    PersonaFactory,
    UserGenerator,
)
from .causal_engine import CausalConfig, CausalEngine
from .scm_generator import SCMConfig, SCMConfounderGenerator
from .time_manager import TimeManager
from .state_manager import StateManager
from .prior_loader import load_priors, get_prior
from .dataset_priors import DatasetPriorExtractor
from .device_utils import (
    detect_device,
    get_torch_device,
    is_gpu_available,
    device_summary,
)
from .causal_structure import (
    CausalStructure,
    CausalStructureConfig,
    CausalStructureSampler,
    get_outcome_equation,
    get_treatment_generator,
    get_mediator_generator,
)
from .aleatory_causal_machine import AleatoryCausalConfig, AleatoryCausalMachine
from .counterfactual_regimes import (
    InterventionRegime,
    get_delta_fn,
    list_available_regimes,
)

__all__ = [
    "IncomeLevel",
    "TechSavviness",
    "PrivacySensitivity",
    "Device",
    "Persona",
    "User",
    "PersonaFactory",
    "UserGenerator",
    "CausalConfig",
    "CausalEngine",
    "SCMConfig",
    "SCMConfounderGenerator",
    "TimeManager",
    "StateManager",
    "load_priors",
    "get_prior",
    "DatasetPriorExtractor",
    "detect_device",
    "get_torch_device",
    "is_gpu_available",
    "device_summary",
    "CausalStructure",
    "CausalStructureConfig",
    "CausalStructureSampler",
    "get_outcome_equation",
    "get_treatment_generator",
    "get_mediator_generator",
    "AleatoryCausalConfig",
    "AleatoryCausalMachine",
    "InterventionRegime",
    "get_delta_fn",
    "list_available_regimes",
]
