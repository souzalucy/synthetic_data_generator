"""Search stream event generation."""

import numpy as np
from typing import List, Dict, Optional
from .base_stream import BaseStream


class SearchStream(BaseStream):
    """Generates search-related events."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        """Initialize search stream with Yandex patterns."""
        super().__init__(config, state_manager, taxonomy, "SEARCH_STREAM", priors)
        self.categories = list(taxonomy["categories"].keys())

        from ecosystem_sim.core.prior_loader import get_prior

        # Load Yandex-specific patterns from priors
        self.yandex_click_position_bias = get_prior(
            priors,
            "layer_2_streams.search.yandex_click_position_bias",
            [0.42, 0.26, 0.15, 0.08, 0.05, 0.02, 0.01, 0.005, 0.003, 0.002],
        )

        self.query_length_dist = get_prior(
            priors,
            "layer_2_streams.search.query_length_distribution",
            {"mean": 3.2, "std": 1.5},
        )

        self.reformulation_rate = get_prior(
            priors, "layer_2_streams.search.reformulation_rate", 0.31
        )

        self.session_length_dist = get_prior(
            priors,
            "layer_2_streams.search.session_length_distribution",
            {"mean": 3.5, "std": 2.1},
        )

        self.no_click_rate = get_prior(
            priors, "layer_2_streams.search.no_click_rate", 0.30
        )
        self.personalization_effect = get_prior(
            priors, "layer_2_streams.search.personalization_effect", 0.28
        )

        # Session state tracking for Yandex-like behavior
        self.user_sessions = {}  # user_id -> session data

        # Standard rates (fallbacks)
        self.serp_click_rate = get_prior(
            priors, "layer_2_streams.search.serp_click_rate", 0.7
        )
        self.organic_click_rate = get_prior(
            priors, "layer_2_streams.search.organic_click_rate", 0.3
        )
        self.ad_impression_rate = get_prior(
            priors, "layer_2_streams.search.ad_impression_rate", 0.4
        )
        self.ad_click_rate = get_prior(
            priors, "layer_2_streams.search.ad_click_rate", None
        )

        # Yandex benchmarks for validation
        self.yandex_benchmarks = {
            "map_baseline": get_prior(
                priors, "layer_2_streams.search.map_baseline", 0.35
            ),
            "mrr_baseline": get_prior(
                priors, "layer_2_streams.search.mrr_baseline", 0.42
            ),
        }

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate search events for a user with Yandex patterns.

        Args:
            user: User object
            timestamp: ISO format timestamp
            probabilities: Action probabilities

        Returns:
            List of events
        """
        events = []

        # Track session for Yandex-like behavior
        session_key = f"{user.user_id}_{timestamp[:10]}"  # user_id + date
        if session_key not in self.user_sessions:
            self.user_sessions[session_key] = {
                "query_count": 0,
                "click_count": 0,
                "last_query_time": None,
                "reformulation_count": 0,
            }

        session = self.user_sessions[session_key]

        # Find search actions in probabilities
        search_actions = [
            k for k in probabilities.keys() if k.startswith("search_query_")
        ]

        for action in search_actions:
            prob = probabilities[action]

            if self.should_generate_event(prob):
                # Extract category index
                cat_idx = int(action.split("_")[-1])
                if cat_idx < len(self.categories):
                    category = self.categories[cat_idx]
                else:
                    category = np.random.choice(self.categories)

                # Select device
                device_id = self.state_manager.get_device_for_action(user, "search")

                # Generate query with Yandex-like patterns
                query = self._generate_yandex_query(category, user, session)

                # Create search_query event
                event = self.create_base_event(
                    user.user_id, device_id, timestamp, "search_query", user=user
                )
                event.update(
                    {
                        "query": query,
                        "category": category,
                        "query_length": len(query.split()),
                        "session_query_num": session["query_count"],
                        "is_reformulation": session["reformulation_count"] > 0,
                    }
                )
                events.append(event)

                session["query_count"] += 1

                # Yandex: 30% of searches have no clicks
                if np.random.random() < self.no_click_rate:
                    # No click event
                    continue

                # Generate SERP view with Yandex click position bias
                serp_event = self.create_base_event(
                    user.user_id, device_id, timestamp, "serp_view", user=user
                )
                serp_event.update(
                    {
                        "query": query,
                        "category": category,
                        "result_count": np.random.randint(5, 100),
                        "personalization_score": min(
                            1.0,
                            self.personalization_effect
                            * user.persona.tech_savviness.value,
                        ),
                    }
                )
                events.append(serp_event)

                # Generate organic clicks with Yandex position bias
                n_clicks = max(
                    1,
                    int(
                        np.random.normal(
                            self.session_length_dist.get("mean", 1.8),
                            self.session_length_dist.get("std", 1.0),
                        )
                    ),
                )
                n_clicks = min(n_clicks, 5)

                for click_idx in range(n_clicks):
                    # Use Yandex click position bias
                    position = self._sample_click_position()
                    click_event = self.create_base_event(
                        user.user_id, device_id, timestamp, "organic_click", user=user
                    )
                    click_event.update(
                        {
                            "query": query,
                            "category": category,
                            "result_position": position,
                            "click_rank": click_idx + 1,
                        }
                    )
                    events.append(click_event)
                    session["click_count"] += 1

                # Ad impression and click with iPinYou-like patterns
                if self.should_generate_event(self.ad_impression_rate):
                    ad_event = self.create_base_event(
                        user.user_id, device_id, timestamp, "ad_impression", user=user
                    )
                    ad_event.update(
                        {
                            "query": query,
                            "category": category,
                            "ad_id": f"ad_{np.random.randint(1000, 9999)}",
                            "ad_position": np.random.choice([1, 2, 3]),  # Top positions
                            "bid_price": np.random.lognormal(
                                mean=0.5, sigma=1.0
                            ),  # iPinYou-like
                        }
                    )
                    events.append(ad_event)

                    # Ad click with persona-adjusted CTR
                    ad_ctr = (
                        user.persona.ad_click_propensity * 0.003
                    )  # Base 0.3% CTR from iPinYou
                    if self.should_generate_event(ad_ctr):
                        ad_click = self.create_base_event(
                            user.user_id, device_id, timestamp, "ad_click", user=user
                        )
                        ad_click.update(
                            {
                                "query": query,
                                "category": category,
                                "ad_id": ad_event["ad_id"],
                                "ad_position": ad_event["ad_position"],
                                "cost_per_click": ad_event["bid_price"]
                                * np.random.uniform(0.8, 1.2),
                            }
                        )
                        events.append(ad_click)

        self.events.extend(events)
        return events

    def _sample_click_position(self) -> int:
        """Sample click position using Yandex click position bias."""
        positions = list(range(1, len(self.yandex_click_position_bias) + 1))
        return np.random.choice(positions, p=self.yandex_click_position_bias)

    def _generate_yandex_query(self, category: str, user, session: Dict) -> str:
        """Generate query with Yandex-like patterns."""
        category_data = self.taxonomy["categories"][category]
        keywords = category_data["keywords"]

        # Sample query length from Yandex distribution
        if isinstance(self.query_length_dist, dict):
            mean_len = self.query_length_dist.get("mean", 3.2)
            std_len = self.query_length_dist.get("std", 1.5)
            n_words = max(1, int(np.random.normal(mean_len, std_len)))
        else:
            n_words = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])

        n_words = min(n_words, len(keywords))
        query_words = np.random.choice(keywords, size=n_words, replace=False)

        query = " ".join(query_words)

        # Check for query reformulation (Yandex pattern)
        if session["query_count"] > 0 and np.random.random() < self.reformulation_rate:
            # Add reformulation markers
            reformulations = ["best", "cheap", "near me", "reviews", "how to"]
            query = query + " " + np.random.choice(reformulations)
            session["reformulation_count"] += 1

        # Add personalization based on user persona
        if user.persona.tech_savviness.value > 0.7:
            tech_modifiers = ["best", "latest", "2024", "top rated", "pro"]
            if np.random.random() < 0.3:
                query = np.random.choice(tech_modifiers) + " " + query

        return query

    def _generate_query(self, category: str, user) -> str:
        """Generate realistic search query.

        Args:
            category: Product category
            user: User object

        Returns:
            Search query string
        """
        # Use Yandex query generation by default
        session_key = f"{user.user_id}_default"
        if session_key not in self.user_sessions:
            self.user_sessions[session_key] = {
                "query_count": 0,
                "click_count": 0,
                "last_query_time": None,
                "reformulation_count": 0,
            }

        return self._generate_yandex_query(
            category, user, self.user_sessions[session_key]
        )
