"""Social stream for social network events."""

import numpy as np
from typing import List, Dict
from .base_stream import BaseStream


class SocialStream(BaseStream):
    """Generates social network events."""
    
    def __init__(self, config: Dict, state_manager, taxonomy: Dict):
        """Initialize social stream."""
        super().__init__(config, state_manager, taxonomy, "SOCIAL_STREAM")
    
    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate social events for a user."""
        events = []
        
        if "social_interaction" not in probabilities:
            return events
        
        # Social interaction
        if self.should_generate_event(probabilities["social_interaction"]):
            # Randomly select a contact
            if user.contact_network:
                contact = np.random.choice(user.contact_network)
                
                event_type = np.random.choice(
                    ["message_sent", "post_created", "reaction_given"],
                    p=[0.4, 0.3, 0.3]
                )
                
                social_event = self.create_base_event(user.user_id, user.devices[0].device_id, timestamp, event_type)
                
                if event_type == "message_sent":
                    social_event.update({
                        "recipient_id": contact,
                        "message_length": np.random.randint(10, 200),
                    })
                elif event_type == "post_created":
                    social_event.update({
                        "post_id": f"post_{np.random.randint(100000, 999999)}",
                        "content_type": np.random.choice(["text", "image", "video"]),
                        "text_length": np.random.randint(20, 500),
                    })
                else:  # reaction_given
                    social_event.update({
                        "post_id": f"post_{np.random.randint(100000, 999999)}",
                        "reaction_type": np.random.choice(["like", "love", "haha", "wow", "sad", "angry"]),
                    })
                
                events.append(social_event)
        
        self.events.extend(events)
        return events
