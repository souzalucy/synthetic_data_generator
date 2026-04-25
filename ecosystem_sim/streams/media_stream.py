"""Media stream for video/content consumption."""

import numpy as np
from typing import List, Dict, Optional
from .base_stream import BaseStream


class MediaStream(BaseStream):
    """Generates media consumption events (video, streaming)."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        """Initialize media stream."""
        super().__init__(config, state_manager, taxonomy, "MEDIA_STREAM", priors)

        from ecosystem_sim.core.prior_loader import get_prior

        self.impression_to_click_rate = get_prior(
            priors, "layer_2_streams.media.impression_to_click_rate", 0.15
        )
        self.video_completion_rate = get_prior(
            priors, "layer_2_streams.media.video_completion_rate", 0.42
        )
        self.category_click_dist = get_prior(
            priors, "layer_2_streams.media.category_click_distribution", None
        )

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate media events for a user."""
        events = []

        # Select best device for media (prefer desktop/tablet)
        media_devices = [
            d for d in user.devices if d.device_type in ["desktop", "tablet"]
        ]
        if not media_devices:
            media_devices = user.devices

        device_id = np.random.choice([d.device_id for d in media_devices])

        if self.should_generate_event(self.impression_to_click_rate):
            if self.category_click_dist:
                cats = list(self.category_click_dist.keys())
                weights = [self.category_click_dist[c] for c in cats]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                chosen_cat = np.random.choice(cats, p=weights)
            else:
                chosen_cat = np.random.choice(list(self.taxonomy["categories"].keys()))

            video_event = self.create_base_event(
                user.user_id, device_id, timestamp, "video_start", user=user
            )
            video_event.update(
                {
                    "video_id": f"vid_{np.random.randint(10000, 99999)}",
                    "category": chosen_cat,
                    "duration_seconds": np.random.choice([180, 300, 600, 1200, 1800]),
                    "quality": np.random.choice(["720p", "1080p", "4K"]),
                }
            )
            events.append(video_event)

            # Watch progress (beta distribution for completion)
            watch_pct = np.random.beta(2, 2)  # More likely to complete
            progress_pcts = [0.25, 0.5, 0.75, 1.0]

            for pct in progress_pcts:
                if watch_pct >= pct and self.should_generate_event(0.8):
                    progress_event = self.create_base_event(
                        user.user_id,
                        device_id,
                        timestamp,
                        f"video_progress_{int(pct * 100)}",
                        user=user,
                    )
                    progress_event.update(
                        {
                            "video_id": video_event["video_id"],
                            "progress_pct": pct,
                        }
                    )
                    events.append(progress_event)

            # Engagement (like, subscribe) if watched enough
            if watch_pct > 0.5:
                if self.should_generate_event(0.2):  # 20% like if watched >50%
                    like_event = self.create_base_event(
                        user.user_id, device_id, timestamp, "video_like", user=user
                    )
                    like_event.update(
                        {
                            "video_id": video_event["video_id"],
                        }
                    )
                    events.append(like_event)

                if self.should_generate_event(0.1):  # 10% subscribe
                    sub_event = self.create_base_event(
                        user.user_id,
                        device_id,
                        timestamp,
                        "channel_subscribe",
                        user=user,
                    )
                    sub_event.update(
                        {
                            "channel_id": f"ch_{np.random.randint(1000, 9999)}",
                        }
                    )
                    events.append(sub_event)

        self.events.extend(events)
        return events
