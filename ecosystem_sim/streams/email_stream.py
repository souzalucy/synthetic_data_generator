"""Email stream for email events."""

import numpy as np
from typing import List, Dict, Optional
from .base_stream import BaseStream


class EmailStream(BaseStream):
    """Generates email-related events."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        """Initialize email stream."""
        super().__init__(config, state_manager, taxonomy, "EMAIL_STREAM", priors)

        from ecosystem_sim.core.prior_loader import get_prior

        self.marketing_email_rate = get_prior(
            priors, "layer_2_streams.email.open_rate_by_type.marketing", 0.22
        )
        self.email_receive_rate = 0.3
        self.spam_rate = get_prior(priors, "layer_2_streams.email.spam_rate", 0.05)
        self.open_rates = get_prior(
            priors,
            "layer_2_streams.email.open_rate_by_type",
            {"marketing": 0.22, "transactional": 0.65, "personal": 0.78},
        )

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate email events for a user."""
        events = []

        # Email reception (marketing emails)
        if self.should_generate_event(self.email_receive_rate):
            email_type = np.random.choice(
                ["marketing", "transactional", "personal"], p=[0.35, 0.35, 0.30]
            )
            email_event = self.create_base_event(
                user.user_id,
                user.devices[0].device_id,
                timestamp,
                "email_received",
                user=user,
            )
            email_event.update(
                {
                    "email_id": f"email_{np.random.randint(100000, 999999)}",
                    "sender": np.random.choice(
                        ["marketing@store.com", "deals@retailer.com", "news@vendor.com"]
                    ),
                    "category": np.random.choice(
                        list(self.taxonomy["categories"].keys())
                    ),
                    "subject": "Exclusive offer"
                    if np.random.random() > 0.5
                    else "New arrivals",
                    "email_type": email_type,
                }
            )
            events.append(email_event)

            open_prob = self.open_rates.get(email_type, 0.3)
            if self.should_generate_event(open_prob):
                open_event = self.create_base_event(
                    user.user_id,
                    user.devices[0].device_id,
                    timestamp,
                    "email_opened",
                    user=user,
                )
                open_event.update(
                    {
                        "email_id": email_event["email_id"],
                    }
                )
                events.append(open_event)

        # Spam filtering
        if self.should_generate_event(self.spam_rate):
            spam_event = self.create_base_event(
                user.user_id,
                user.devices[0].device_id,
                timestamp,
                "spam_filtered",
                user=user,
            )
            spam_event.update(
                {
                    "email_id": f"spam_{np.random.randint(100000, 999999)}",
                }
            )
            events.append(spam_event)

        return events
