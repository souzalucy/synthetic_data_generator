"""Propensity models for prediction."""

import numpy as np
from typing import List, Dict


class PropensityModels:
    """Calculate propensity scores (LTV, Churn, Conversion)."""
    
    def __init__(self):
        """Initialize propensity models."""
        self.ltv_model = None
        self.churn_model = None
        self.conversion_model = None
    
    def calculate_ltv(self, user, events: List[Dict], user_profile: Dict) -> float:
        """Calculate Lifetime Value prediction.
        
        Args:
            user: User object
            events: List of user events
            user_profile: User profile dictionary
            
        Returns:
            LTV in dollars
        """
        # Historical spend
        purchases = [e for e in events if e.get("event_type") == "purchase"]
        total_spend = sum(e.get("price_usd", 0) for e in purchases)
        
        # Engagement score
        engagement = len(events) / max(1, len(set()))  # Normalize
        
        # Persona multiplier
        persona_mult = {
            "Business Professional": 2.0,
            "Tech-Savvy Millennial": 1.5,
            "Budget-Conscious Parent": 1.0,
            "Casual Browser": 0.5,
            "Privacy-Focused User": 0.8,
        }.get(user.persona.name, 1.0)
        
        # Project future value
        base_ltv = total_spend if total_spend > 0 else 100
        ltv = base_ltv * (1 + engagement * 0.5) * persona_mult
        
        return max(0, ltv)
    
    def calculate_churn_risk(self, user, events: List[Dict]) -> float:
        """Calculate churn risk (probability of becoming inactive).
        
        Args:
            user: User object
            events: List of user events
            
        Returns:
            Churn risk [0, 1]
        """
        if not events:
            return 0.9  # New users have no history
        
        # Recency (days since last event)
        from datetime import datetime
        last_event_time = max(
            datetime.fromisoformat(e.get("timestamp", "").replace('Z', '+00:00'))
            for e in events if e.get("timestamp")
        )
        days_inactive = (datetime.now(last_event_time.tzinfo) - last_event_time).days
        
        # Frequency (events per day)
        first_event_time = min(
            datetime.fromisoformat(e.get("timestamp", "").replace('Z', '+00:00'))
            for e in events if e.get("timestamp")
        )
        account_age_days = (last_event_time - first_event_time).days + 1
        frequency = len(events) / max(1, account_age_days)
        
        # Base churn risk
        churn = max(0.1, min(0.9, 0.5 * (days_inactive / 30) - 0.3 * frequency))
        
        # Persona adjustment
        if user.persona.name == "Tech-Savvy Millennial":
            churn *= 0.7
        elif user.persona.name == "Casual Browser":
            churn *= 1.3
        
        return np.clip(churn, 0, 1)
    
    def calculate_conversion_propensity(self, user, events: List[Dict]) -> float:
        """Calculate probability of purchase in next 30 days.
        
        Args:
            user: User object
            events: List of user events
            
        Returns:
            Conversion propensity [0, 1]
        """
        if not events:
            return user.persona.impulse_buying * 0.5
        
        # Purchase frequency
        purchases = [e for e in events if e.get("event_type") == "purchase"]
        purchase_rate = len(purchases) / max(1, len(events))
        
        # Browse/search activity
        browse_events = [e for e in events if e.get("event_type") in 
                         ["search_query", "product_view", "add_to_cart"]]
        browse_rate = len(browse_events) / max(1, len(events))
        
        # Propensity
        propensity = (
            0.3 * purchase_rate +
            0.3 * browse_rate +
            0.4 * user.persona.impulse_buying
        )
        
        return np.clip(propensity, 0, 1)
    
    def calculate_all(self, stitched_data: Dict) -> Dict:
        """Calculate all propensity scores.
        
        Args:
            stitched_data: Stitched event data
            
        Returns:
            Dictionary of propensity scores
        """
        # Placeholder: would aggregate events by user_id
        return {
            "ltv_scores": {},
            "churn_risks": {},
            "conversion_propensities": {},
        }
