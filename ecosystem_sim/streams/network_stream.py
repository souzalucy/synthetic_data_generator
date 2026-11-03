"""Network interaction stream for GitHub, Reddit, and other platform interactions."""

import numpy as np
from typing import List, Dict, Optional
from .base_stream import BaseStream


class NetworkStream(BaseStream):
    """Generates network interaction events (GitHub stars, Reddit posts, etc.)."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        super().__init__(config, state_manager, taxonomy, "NETWORK_STREAM", priors)

        from ecosystem_sim.core.prior_loader import get_prior

        # Load network priors
        self.github_priors = get_prior(
            priors, "layer_2_streams.network.github_interactions", {}
        )

        self.reddit_priors = get_prior(
            priors, "layer_2_streams.network.reddit_discussions", {}
        )

        self.platform_engagement = get_prior(
            priors,
            "layer_2_streams.network.platform_engagement_rates",
            {
                "github_star_rate": 0.08,
                "reddit_post_rate": 0.12,
                "product_review_rate": 0.05,
                "code_contribution_rate": 0.03,
            },
        )

        self.subreddit_affinity = get_prior(
            priors,
            "layer_2_streams.network.subreddit_affinity_matrix",
            {
                "programming→gaming": 0.25,
                "gaming→movies": 0.15,
                "movies→television": 0.20,
                "television→news": 0.18,
            },
        )

        # Developer activity levels
        self.developer_types = get_prior(
            priors,
            "layer_1_persona.developer_activity_levels.developer_types",
            {
                "inactive": 0.6,
                "casual": 0.3,
                "active": 0.08,
                "core": 0.02,
            },
        )

        # Initialize user developer profiles
        self.user_developer_profiles = {}

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate network interaction events."""
        events = []

        # Determine if user is a developer
        is_developer = self._is_developer(user)

        # GitHub events (only for developers)
        if is_developer and self.should_generate_event(
            probabilities.get("github", 0.1)
        ):
            events.append(self._create_github_event(user, timestamp))

        # Reddit events (for all users)
        if self.should_generate_event(probabilities.get("reddit", 0.15)):
            events.append(self._create_reddit_event(user, timestamp))

        # Product review events
        if self.should_generate_event(probabilities.get("review", 0.05)):
            events.append(self._create_product_review_event(user, timestamp))

        return events

    def _is_developer(self, user) -> bool:
        """Determine if user is a developer based on persona."""
        user_id = user.user_id

        if user_id not in self.user_developer_profiles:
            # Check if persona has developer flag
            if hasattr(user, "persona") and user.persona:
                # Persona may be a dict-like object or a Persona instance
                if hasattr(user.persona, "get"):
                    is_dev = user.persona.get("is_developer", False)
                else:
                    # Assume Persona object, check for is_developer attribute
                    is_dev = getattr(user.persona, "is_developer", False)
            else:
                # Random assignment based on population ratio
                dev_ratio = (
                    self.config.get("streams", {})
                    .get("network", {})
                    .get("developer_population_ratio", 0.15)
                )
                is_dev = np.random.random() < dev_ratio

            # Assign developer type
            if is_dev:
                dev_types = list(self.developer_types.keys())
                dev_probs = list(self.developer_types.values())
                dev_type = np.random.choice(dev_types, p=dev_probs)
            else:
                dev_type = "non_developer"

            self.user_developer_profiles[user_id] = {
                "is_developer": is_dev,
                "developer_type": dev_type,
                "github_repos": [] if is_dev else None,
                "reddit_subscriptions": [],
            }

        return self.user_developer_profiles[user_id]["is_developer"]

    def _create_github_event(self, user, timestamp: str) -> Dict:
        """Create GitHub interaction event."""
        device_id = user.devices[0].device_id if user.devices else "unknown"
        base_event = self.create_base_event(
            user.user_id, device_id, timestamp, "GITHUB_INTERACTION", user=user
        )

        profile = self.user_developer_profiles[user.user_id]
        dev_type = profile["developer_type"]

        # Determine action based on developer type
        if dev_type == "inactive":
            actions = ["view_repo", "read_issue"]
            probs = [0.7, 0.3]
        elif dev_type == "casual":
            actions = ["star_repo", "fork_repo", "create_issue"]
            probs = [0.5, 0.3, 0.2]
        elif dev_type == "active":
            actions = ["create_pull_request", "merge_pull_request", "commit_code"]
            probs = [0.4, 0.3, 0.3]
        else:  # core
            actions = ["create_repo", "review_pull_request", "deploy_release"]
            probs = [0.3, 0.4, 0.3]

        action = np.random.choice(actions, p=probs)

        # Programming languages
        languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript"]
        language = np.random.choice(
            languages, p=[0.35, 0.25, 0.15, 0.08, 0.07, 0.05, 0.05]
        )

        return {
            **base_event,
            "action": action,
            "platform": "github",
            "repository_language": language,
            "developer_type": dev_type,
            "is_open_source": np.random.random() < 0.8,
            "lines_of_code": int(np.random.exponential(50))
            if action in ["commit_code", "create_pull_request"]
            else 0,
            "issue_complexity": np.random.choice(
                ["low", "medium", "high"], p=[0.6, 0.3, 0.1]
            ),
        }

    def _create_reddit_event(self, user, timestamp: str) -> Dict:
        """Create Reddit interaction event."""
        device_id = user.devices[0].device_id if user.devices else "unknown"
        base_event = self.create_base_event(
            user.user_id, device_id, timestamp, "REDDIT_INTERACTION", user=user
        )

        # Subreddit categories
        subreddits = {
            "programming": ["r/programming", "r/python", "r/javascript", "r/webdev"],
            "gaming": ["r/gaming", "r/pcgaming", "r/indiegames"],
            "movies": ["r/movies", "r/television", "r/netflix"],
            "news": ["r/news", "r/worldnews", "r/politics"],
            "technology": ["r/technology", "r/futurology", "r/cybersecurity"],
        }

        # Choose category based on user interests
        if hasattr(user, "persona") and user.persona:
            # Persona may be a dict-like object or a Persona instance
            if hasattr(user.persona, "get"):
                interests = user.persona.get("interests", [])
            else:
                interests = getattr(user.persona, "interests", [])
            if "technology" in interests:
                category = "programming"
            elif "gaming" in interests:
                category = "gaming"
            elif "entertainment" in interests:
                category = "movies"
            else:
                categories = list(subreddits.keys())
                category = np.random.choice(categories)
        else:
            categories = list(subreddits.keys())
            category = np.random.choice(categories)

        subreddit = np.random.choice(subreddits[category])

        # Interaction types
        actions = ["view_post", "upvote", "downvote", "comment", "create_post"]
        action_probs = [0.4, 0.25, 0.05, 0.2, 0.1]
        action = np.random.choice(actions, p=action_probs)

        # Generate post/comment content metrics
        word_count = 0
        if action in ["comment", "create_post"]:
            word_count = int(
                np.random.lognormal(3.0, 0.8)
            )  # Most comments 20-100 words

        return {
            **base_event,
            "action": action,
            "platform": "reddit",
            "subreddit": subreddit,
            "category": category,
            "post_word_count": word_count,
            "has_media": np.random.random() < 0.3,
            "sentiment_score": float(np.random.uniform(-0.5, 0.8)),  # -0.5 to 0.8
            "controversial_score": float(
                np.random.beta(2, 5)
            ),  # Mostly low controversy
        }

    def _create_product_review_event(self, user, timestamp: str) -> Dict:
        """Create product review event."""
        device_id = user.devices[0].device_id if user.devices else "unknown"
        base_event = self.create_base_event(
            user.user_id, device_id, timestamp, "PRODUCT_REVIEW", user=user
        )

        # Product categories
        categories = ["electronics", "books", "home", "clothing", "software", "games"]
        category = np.random.choice(categories, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])

        # Review rating (1-5 stars, biased toward positive)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.4, 0.3])

        # Review helpfulness
        helpful_votes = int(np.random.poisson(3))
        total_votes = helpful_votes + int(np.random.poisson(1))

        return {
            **base_event,
            "action": "write_review",
            "platform": "amazon",  # or other e-commerce
            "product_category": category,
            "rating": rating,
            "review_word_count": int(
                np.random.lognormal(3.5, 0.7)
            ),  # ~30 words average
            "has_photos": np.random.random() < 0.4,
            "verified_purchase": np.random.random() < 0.7,
            "helpful_votes": helpful_votes,
            "total_votes": total_votes,
            "helpfulness_score": helpful_votes / max(total_votes, 1),
        }

    def should_generate_event(self, probability: float) -> bool:
        """Determine if an event should be generated."""
        return np.random.random() < probability
