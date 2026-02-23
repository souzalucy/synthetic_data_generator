"""State manager bridging causal engine and time manager."""

from typing import Dict, List
import numpy as np
from .causal_engine import CausalEngine
from .time_manager import TimeManager
from .user_agent import User


class StateManager:
    """Bridges causal interests and temporal patterns to compute action probabilities."""
    
    def __init__(self, causal_engine: CausalEngine, time_manager: TimeManager, 
                 interest_matrix: np.ndarray):
        """Initialize state manager.
        
        Args:
            causal_engine: Causal inference engine
            time_manager: Time pattern manager
            interest_matrix: Pre-computed interest trajectories
        """
        self.causal_engine = causal_engine
        self.time_manager = time_manager
        self.interest_matrix = interest_matrix
        self.in_market_threshold = 0.7
    
    def get_daily_action_probabilities(self, user: User, day_index: int, 
                                       hour: int, users: List[User]) -> Dict[str, float]:
        """Get action probabilities for a user at a specific time.
        
        This is the critical bridging function that connects:
        - Causal interests (from interest_matrix)
        - Temporal patterns (from time_manager)
        - Persona characteristics (from user)
        
        Args:
            user: User object
            day_index: Day in simulation (0-indexed)
            hour: Hour of day (0-23)
            users: List of all users (for indexing)
            
        Returns:
            Dictionary mapping action names to probabilities [0, 1]
        """
        # Find user index
        user_idx = next(i for i, u in enumerate(users) if u.user_id == user.user_id)
        
        # Get causal interest at this day
        day_interests = self.interest_matrix[user_idx, day_index, :]
        
        # Get day of week and month
        day_of_week = self.time_manager.get_day_of_week()
        month = self.time_manager.get_month()
        
        # Compute temporal multipliers for each service
        time_mults = {}
        for service in ["search", "commerce", "geo", "media", "social", "email"]:
            time_mults[service] = self.time_manager.get_combined_multiplier(
                service, hour, day_of_week, month
            )
        
        # Base persona modifier
        persona_search_freq = user.persona.base_search_frequency / 5.0  # Normalize to [0, 1]
        persona_commerce_freq = user.persona.impulse_buying
        
        # Build action probabilities
        probabilities = {}
        
        # Search actions
        for cat_idx, interest in enumerate(day_interests):
            if interest > self.in_market_threshold:
                action_name = f"search_query_{cat_idx}"
                prob = (interest * time_mults["search"] * 
                       persona_search_freq * user.social_connectivity_score)
                probabilities[action_name] = np.clip(prob, 0, 1)
        
        # Commerce actions (subset of search)
        for cat_idx, interest in enumerate(day_interests):
            # Commerce requires higher interest AND persona propensity
            commerce_threshold = self.in_market_threshold * 1.2
            if interest > commerce_threshold and interest > day_interests.mean():
                action_name = f"commerce_browse_{cat_idx}"
                prob = (interest * time_mults["commerce"] * 
                       persona_commerce_freq * 0.3)
                probabilities[action_name] = np.clip(prob, 0, 1)
        
        # Ad click actions (subset of search)
        for cat_idx, interest in enumerate(day_interests):
            if interest > self.in_market_threshold:
                action_name = f"ad_click_{cat_idx}"
                prob = (interest * user.persona.ad_click_propensity * 
                       time_mults["media"] * 0.1)
                probabilities[action_name] = np.clip(prob, 0, 1)
        
        # Geo events (mobile only)
        if any(d.device_type == "mobile" for d in user.devices):
            probabilities["geo_update"] = time_mults["geo"] * 0.5
        
        # Social events
        probabilities["social_interaction"] = (
            time_mults["social"] * 
            user.social_connectivity_score * 0.2
        )
        
        # Email events
        probabilities["email_receive"] = time_mults["email"] * 0.2
        
        return probabilities
    
    def is_in_market(self, user: User, day_index: int, 
                    users: List[User]) -> bool:
        """Check if user is in-market (high interest) on a given day.
        
        Args:
            user: User object
            day_index: Day in simulation
            users: List of all users
            
        Returns:
            True if mean interest > threshold
        """
        user_idx = next(i for i, u in enumerate(users) if u.user_id == user.user_id)
        mean_interest = self.interest_matrix[user_idx, day_index, :].mean()
        return mean_interest > self.in_market_threshold
    
    def get_device_for_action(self, user: User, service: str) -> str:
        """Select a device for an action based on service and persona.
        
        Args:
            user: User object
            service: Service name
            
        Returns:
            Selected device_id
        """
        if len(user.devices) == 1:
            return user.devices[0].device_id
        
        # Preference by service
        if service in ["search", "commerce"]:
            # 70% mobile, 30% desktop for most users
            if user.persona.device_pref_mobile > 0.7 or np.random.random() < 0.7:
                mobile_devices = [d for d in user.devices if d.device_type == "mobile"]
                if mobile_devices:
                    return np.random.choice([d.device_id for d in mobile_devices])
        
        if service == "geo":
            # Geo is mobile-only
            mobile_devices = [d for d in user.devices if d.device_type == "mobile"]
            if mobile_devices:
                return np.random.choice([d.device_id for d in mobile_devices])
            else:
                return user.devices[0].device_id
        
        if service == "media":
            # Media can be any device, slightly favor desktop for longer content
            if np.random.random() < 0.4:
                desktop_devices = [d for d in user.devices if d.device_type == "desktop"]
                if desktop_devices:
                    return np.random.choice([d.device_id for d in desktop_devices])
        
        # Default: random device
        return np.random.choice([d.device_id for d in user.devices])
