"""Email stream for email events."""

import numpy as np
from typing import List, Dict
from .base_stream import BaseStream


class EmailStream(BaseStream):
    """Generates email-related events."""
    
    def __init__(self, config: Dict, state_manager, taxonomy: Dict):
        """Initialize email stream."""
        super().__init__(config, state_manager, taxonomy, "EMAIL_STREAM")
    
    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate email events for a user."""
        events = []
        
        # Email reception (marketing emails)
        if self.should_generate_event(0.3):  # 30% chance of email
            email_event = self.create_base_event(user.user_id, user.devices[0].device_id, timestamp, "email_received")
            email_event.update({
                "email_id": f"email_{np.random.randint(100000, 999999)}",
                "sender": np.random.choice(["marketing@store.com", "deals@retailer.com", "news@vendor.com"]),
                "category": np.random.choice(list(self.taxonomy["categories"].keys())),
                "subject": "Exclusive offer" if np.random.random() > 0.5 else "New arrivals",
            })
            events.append(email_event)
            
            # Email open (proportional to persona)
            open_prob = 0.4 if user.persona.ad_click_propensity > 0.3 else 0.2
            if self.should_generate_event(open_prob):
                open_event = self.create_base_event(user.user_id, user.devices[0].device_id, timestamp, "email_opened")
                open_event.update({
                    "email_id": email_event["email_id"],
                })
                events.append(open_event)
        
        # Spam filtering
        if self.should_generate_event(0.05):  # 5% spam
            spam_event = self.create_base_event(user.user_id, user.devices[0].device_id, timestamp, "spam_filtered")
            spam_event.update({
                "email_id": f"spam_{np.random.randint(100000, 999999)}",
            })
            events.append(spam_event)
        
        self.events.extend(events)
        return events
