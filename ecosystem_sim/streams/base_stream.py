"""Base stream class for event generation."""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod


class BaseStream(ABC):
    """Abstract base class for all event streams."""
    
    def __init__(self, config: Dict, state_manager, taxonomy: Dict, name: str):
        """Initialize base stream.
        
        Args:
            config: Simulation configuration
            state_manager: State manager instance
            taxonomy: Taxonomy dictionary
            name: Stream name
        """
        self.config = config
        self.state_manager = state_manager
        self.taxonomy = taxonomy
        self.name = name
        self.events = []
        self.event_counter = 0
    
    def create_base_event(self, user_id: str, device_id: str, 
                         timestamp: str, event_type: str) -> Dict[str, Any]:
        """Create base event structure.
        
        Args:
            user_id: User ID
            device_id: Device ID
            timestamp: ISO format timestamp
            event_type: Type of event
            
        Returns:
            Base event dictionary
        """
        self.event_counter += 1
        
        # Find device to get device_type and os
        device_info = {"device_type": "unknown", "os": "unknown"}
        
        return {
            "event_id": f"evt_{self.event_counter:010d}_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "device_id": device_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "service": self.name,
            **device_info
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
        with open(filepath, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event) + '\n')
    
    def clear_events(self):
        """Clear in-memory events (for memory management)."""
        self.events = []
