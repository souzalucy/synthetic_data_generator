"""Base stream class for event generation."""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


class BaseStream(ABC):
    """Abstract base class for all event streams."""

    def __init__(
        self,
        config: Dict,
        state_manager,
        taxonomy: Dict,
        name: str,
        priors: Optional[Dict] = None,
    ):
        """Initialize base stream.

        Args:
            config: Simulation configuration
            state_manager: State manager instance
            taxonomy: Taxonomy dictionary
            name: Stream name
            priors: Optional compiled dataset priors
        """
        self.config = config
        self.state_manager = state_manager
        self.taxonomy = taxonomy
        self.name = name
        self.priors = priors
        self.events = []
        self.event_counter = 0

    def create_base_event(
        self,
        user_id: str,
        device_id: str,
        timestamp: str,
        event_type: str,
        user: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Create base event structure.

        Args:
            user_id: User ID (may be omitted probabilistically based on privacy)
            device_id: Device ID
            timestamp: ISO format timestamp
            event_type: Type of event
            user: Optional User object for privacy-based user_id omission

        Returns:
            Base event dictionary
        """
        self.event_counter += 1

        # Find device to get device_type and os
        device_info = {"device_type": "unknown", "os": "unknown"}

        # Probabilistic user_id omission based on privacy sensitivity
        # Privacy-Focused users have HIGH sensitivity -> more likely to omit user_id
        # This creates realistic unidentified logs for the GraphStitcher to resolve
        resolved_user_id = user_id
        if user is not None:
            from ecosystem_sim.core.user_agent import PrivacySensitivity

            privacy_val = (
                user.persona.privacy_sensitivity.value
            )  # LOW=0.2, MEDIUM=0.5, HIGH=0.9
            if np.random.random() < privacy_val:
                resolved_user_id = None

        return {
            "event_id": f"evt_{self.event_counter:010d}_{uuid.uuid4().hex[:8]}",
            "user_id": resolved_user_id,
            "device_id": device_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "service": self.name,
            **device_info,
        }

    def should_generate_event(self, probability: float) -> bool:
        """Stochastic decision for event generation.

        Args:
            probability: Probability [0, 1]

        Returns:
            True if event should be generated
        """
        import numpy as np

        return np.random.random() < probability

    @abstractmethod
    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate event(s) for a user at a timestamp.

        Must be implemented by subclasses.

        Args:
            user: User object
            timestamp: ISO format timestamp
            probabilities: Action probabilities from state manager

        Returns:
            List of events (may be empty)
        """
        pass

    def export_events(self, filepath: str):
        """Export events to JSONL file.

        Args:
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            for event in self.events:
                f.write(json.dumps(event, cls=NumpyEncoder) + "\n")

    def clear_events(self):
        """Clear in-memory events (for memory management)."""
        self.events = []
