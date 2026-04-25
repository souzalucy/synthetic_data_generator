"""Social stream for social network events with GoEmotions sentiment dynamics."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .base_stream import BaseStream
from datetime import datetime, timedelta
import random


class SocialStream(BaseStream):
    """Generates social network events with GoEmotions sentiment dynamics."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        """Initialize social stream with GoEmotions patterns."""
        super().__init__(config, state_manager, taxonomy, "SOCIAL_STREAM", priors)

        from ecosystem_sim.core.prior_loader import get_prior

        self.interaction_dist = get_prior(
            priors, "layer_2_streams.social.interaction_type_distribution", None
        )
        self.emotion_dist = get_prior(
            priors, "layer_2_streams.social.emotion_distribution", None
        )
        self.ekman_mapping = get_prior(
            priors, "layer_2_streams.social.ekman_mapping", None
        )
        self.sentiment_mapping = get_prior(
            priors, "layer_2_streams.social.sentiment_mapping", None
        )

        # User emotion states
        self.user_emotion_states = {}

        # GoEmotions 27 emotion categories (simplified mapping)
        self.goemotions_categories = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]

        # Ekman basic emotions mapping
        self.ekman_basic = {
            "anger": ["anger", "annoyance", "disapproval", "disgust"],
            "disgust": ["disgust"],
            "fear": ["fear", "nervousness"],
            "joy": [
                "joy",
                "amusement",
                "excitement",
                "gratitude",
                "love",
                "pride",
                "relief",
            ],
            "sadness": [
                "sadness",
                "disappointment",
                "embarrassment",
                "grief",
                "remorse",
            ],
            "surprise": ["surprise", "realization"],
            "neutral": [
                "neutral",
                "approval",
                "caring",
                "confusion",
                "curiosity",
                "desire",
                "admiration",
                "optimism",
            ],
        }

        # Sentiment valence mapping
        self.sentiment_valence = {
            "positive": [
                "admiration",
                "amusement",
                "approval",
                "caring",
                "excitement",
                "gratitude",
                "joy",
                "love",
                "optimism",
                "pride",
                "relief",
            ],
            "negative": [
                "anger",
                "annoyance",
                "confusion",
                "disappointment",
                "disapproval",
                "disgust",
                "embarrassment",
                "fear",
                "grief",
                "nervousness",
                "remorse",
                "sadness",
            ],
            "neutral": ["curiosity", "desire", "realization", "surprise", "neutral"],
        }

        # Emotion transition probabilities (simplified Markov model)
        self.emotion_transitions = self._build_emotion_transition_matrix()

        # Social contagion parameters
        self.contagion_strength = 0.3  # How strongly emotions spread
        self.contagion_decay = 0.8  # How quickly contagion effect decays

        # Initialize interaction types
        if self.interaction_dist:
            self._int_types = list(self.interaction_dist.keys())
            self._int_probs = [self.interaction_dist[t] for t in self._int_types]
            total = sum(self._int_probs)
            self._int_probs = [p / total for p in self._int_probs]
            self._type_mapping = {
                "message": "message_sent",
                "post": "post_created",
                "reaction": "reaction_given",
                "comment": "comment_posted",
                "share": "content_shared",
                "mention": "user_mentioned",
            }
        else:
            self._int_types = None

        # Initialize emotion distribution
        if self.emotion_dist:
            self._emotions = list(self.emotion_dist.keys())
            self._emotion_probs = [self.emotion_dist[e] for e in self._emotions]
            total = sum(self._emotion_probs)
            self._emotion_probs = [p / total for p in self._emotion_probs]
        else:
            # Default uniform distribution over GoEmotions categories
            self._emotions = self.goemotions_categories
            self._emotion_probs = [1.0 / len(self._emotions)] * len(self._emotions)

    def _build_emotion_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build Markov transition matrix for emotion dynamics."""
        transitions = {}

        for emotion in self.goemotions_categories:
            transitions[emotion] = {}

            # Self-transition weight
            transitions[emotion][emotion] = 0.4

            # Compute similarity weights for other emotions
            other_emotions = [e for e in self.goemotions_categories if e != emotion]
            raw_weights = [self._emotion_similarity(emotion, e) for e in other_emotions]
            total_raw = sum(raw_weights)

            # Distribute remaining 0.6 proportionally to similarity,
            # guaranteeing the row sums to 1.0
            for other, weight in zip(other_emotions, raw_weights):
                transitions[emotion][other] = 0.6 * weight / total_raw

        return transitions

    def _emotion_similarity(self, emotion1: str, emotion2: str) -> float:
        """Calculate similarity between two emotions."""
        # Check if in same Ekman category
        for ekman, emotions in self.ekman_basic.items():
            if emotion1 in emotions and emotion2 in emotions:
                return 0.8

        # Check if in same sentiment valence
        for valence, emotions in self.sentiment_valence.items():
            if emotion1 in emotions and emotion2 in emotions:
                return 0.5

        # Default low similarity
        return 0.1

    def _initialize_user_emotion_state(self, user_id: str) -> Dict:
        """Initialize user's emotion state."""
        # Start with neutral or random emotion
        initial_emotion = np.random.choice(
            self.goemotions_categories, p=self._emotion_probs
        )

        return {
            "current_emotion": initial_emotion,
            "emotion_intensity": np.random.uniform(0.3, 0.8),
            "last_emotion_change": None,
            "emotion_history": [],
            "social_influence": {},  # Influence from other users
            "persona_modifier": self._get_persona_emotion_modifier(user_id),
        }

    def _get_persona_emotion_modifier(self, user_id: str) -> float:
        """Get emotion stability modifier based on user persona."""
        # In a real implementation, this would check user persona
        # For now, return random modifier
        return np.random.uniform(0.8, 1.2)

    def _update_emotion_state(
        self, user_id: str, timestamp: str, social_context: Optional[Dict] = None
    ) -> str:
        """Update user's emotion state based on time and social context."""
        if user_id not in self.user_emotion_states:
            self.user_emotion_states[user_id] = self._initialize_user_emotion_state(
                user_id
            )

        state = self.user_emotion_states[user_id]
        current_emotion = state["current_emotion"]

        # Apply emotion transitions
        if np.random.random() < 0.2:  # 20% chance of emotion change per event
            transition_probs = self.emotion_transitions[current_emotion]
            emotions = list(transition_probs.keys())
            probs = list(transition_probs.values())

            new_emotion = np.random.choice(emotions, p=probs)
            state["current_emotion"] = new_emotion
            state["emotion_intensity"] = np.random.uniform(0.3, 0.8)
            state["last_emotion_change"] = timestamp

            # Record in history
            state["emotion_history"].append(
                {
                    "timestamp": timestamp,
                    "emotion": new_emotion,
                    "intensity": state["emotion_intensity"],
                    "context": social_context,
                }
            )

            # Keep history manageable
            if len(state["emotion_history"]) > 100:
                state["emotion_history"] = state["emotion_history"][-50:]

        # Apply social contagion if context provided
        if social_context and "interacted_with" in social_context:
            interacted_user = social_context["interacted_with"]
            if interacted_user in self.user_emotion_states:
                other_state = self.user_emotion_states[interacted_user]
                other_emotion = other_state["current_emotion"]

                # Contagion effect decays with time
                contagion_prob = self.contagion_strength * state["persona_modifier"]
                if np.random.random() < contagion_prob:
                    # Partially adopt other's emotion
                    if np.random.random() < 0.3:
                        state["current_emotion"] = other_emotion
                        state["emotion_intensity"] = (
                            state["emotion_intensity"] * 0.7
                            + other_state["emotion_intensity"] * 0.3
                        )

        return state["current_emotion"]

    def _generate_social_content(self, emotion: str, event_type: str) -> Dict:
        """Generate social content based on emotion and event type."""
        content = {}

        # Text templates based on emotion
        emotion_templates = {
            "joy": [
                "Having a great day! 😊",
                "So excited about this!",
                "This made me really happy!",
                "Feeling blessed today 🙏",
            ],
            "sadness": [
                "Not having the best day...",
                "Feeling a bit down today",
                "Could use some positivity",
                "Going through a tough time",
            ],
            "anger": [
                "This is so frustrating!",
                "Why does this always happen?",
                "I've had enough of this",
                "Seriously annoyed right now",
            ],
            "surprise": [
                "Wow, didn't see that coming!",
                "This is unexpected!",
                "Can't believe this happened",
                "What a surprise!",
            ],
            "neutral": [
                "Just checking in",
                "Another day, another dollar",
                "Going about my day",
                "Nothing special today",
            ],
        }

        # Map GoEmotion to basic emotion for templates
        basic_emotion = "neutral"
        for ekman, emotions in self.ekman_basic.items():
            if emotion in emotions:
                basic_emotion = ekman
                break

        if event_type == "post_created":
            templates = emotion_templates.get(
                basic_emotion, emotion_templates["neutral"]
            )
            content["text"] = random.choice(templates)
            content["hashtags"] = self._generate_hashtags(emotion)
            content["mentions"] = [] if np.random.random() < 0.7 else ["@friend"]

        elif event_type == "message_sent":
            greetings = ["Hey", "Hi", "Hello", "Hey there"]
            content["text"] = f"{random.choice(greetings)}, how are you?"

        elif event_type == "comment_posted":
            comments = [
                "Nice!",
                "Interesting",
                "I agree",
                "Good point",
                "Thanks for sharing",
            ]
            content["text"] = random.choice(comments)

        return content

    def _generate_hashtags(self, emotion: str) -> List[str]:
        """Generate relevant hashtags based on emotion."""
        emotion_hashtags = {
            "joy": ["#happy", "#blessed", "#goodvibes", "#positive"],
            "sadness": ["#feelingdown", "#hardtimes", "#needhugs"],
            "anger": ["#frustrated", "#annoyed", "#venting"],
            "excitement": ["#excited", "#cantwait", "#thrilled"],
            "love": ["#love", "#relationshipgoals", "#caring"],
            "neutral": ["#daily", "#life", "#update"],
        }

        # Find matching hashtags
        hashtags = []
        for basic_emotion, tags in emotion_hashtags.items():
            if emotion in self.ekman_basic.get(basic_emotion, []):
                hashtags.extend(tags[:2])

        if not hashtags:
            hashtags = ["#life", "#update"]

        return hashtags

    def _get_ekman_category(self, emotion: str) -> Optional[str]:
        """Map GoEmotion to Ekman basic emotion category."""
        for ekman, emotions in self.ekman_basic.items():
            if emotion in emotions:
                return ekman
        return None

    def _get_sentiment_valence(self, emotion: str) -> str:
        """Get sentiment valence (positive/negative/neutral) for emotion."""
        for valence, emotions in self.sentiment_valence.items():
            if emotion in emotions:
                return valence
        return "neutral"

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate social events with GoEmotions sentiment dynamics."""
        events = []

        if "social_interaction" not in probabilities:
            return events

        # Social interaction
        if self.should_generate_event(probabilities["social_interaction"]):
            # Update user's emotion state
            social_context = {}

            # Randomly select a contact if available
            contact = None
            if user.contact_network:
                contact = np.random.choice(user.contact_network)
                social_context["interacted_with"] = contact

            current_emotion = self._update_emotion_state(
                user.user_id, timestamp, social_context
            )

            # Determine interaction type
            if self._int_types:
                raw_type = np.random.choice(self._int_types, p=self._int_probs)
                event_type = self._type_mapping.get(raw_type, "message_sent")
            else:
                event_type = np.random.choice(
                    [
                        "message_sent",
                        "post_created",
                        "reaction_given",
                        "comment_posted",
                    ],
                    p=[0.4, 0.3, 0.2, 0.1],
                )

            # Get sentiment information
            ekman_category = self._get_ekman_category(current_emotion)
            sentiment_valence = self._get_sentiment_valence(current_emotion)

            # Generate social content
            content = self._generate_social_content(current_emotion, event_type)

            # Create base event
            device_id = user.devices[0].device_id if user.devices else "unknown"
            social_event = self.create_base_event(
                user.user_id, device_id, timestamp, event_type, user=user
            )

            # Add event-specific fields
            if event_type == "message_sent":
                social_event.update(
                    {
                        "recipient_id": contact if contact else "unknown",
                        "message_length": len(content.get("text", "")),
                        "message_text": content.get("text", ""),
                        "has_attachment": np.random.random() < 0.1,
                    }
                )
            elif event_type == "post_created":
                social_event.update(
                    {
                        "post_id": f"post_{np.random.randint(100000, 999999)}",
                        "content_type": np.random.choice(
                            ["text", "image", "video", "link"]
                        ),
                        "text_length": len(content.get("text", "")),
                        "post_text": content.get("text", ""),
                        "hashtags": content.get("hashtags", []),
                        "mentions": content.get("mentions", []),
                        "is_public": np.random.random() < 0.8,
                    }
                )
            elif event_type == "reaction_given":
                # Reaction types weighted by emotion
                reaction_weights = {
                    "like": 0.4,
                    "love": 0.2,
                    "haha": 0.15,
                    "wow": 0.1,
                    "sad": 0.1,
                    "angry": 0.05,
                }

                # Adjust weights based on emotion
                if sentiment_valence == "positive":
                    reaction_weights["love"] += 0.1
                    reaction_weights["like"] += 0.1
                elif sentiment_valence == "negative":
                    reaction_weights["sad"] += 0.1
                    reaction_weights["angry"] += 0.1

                reactions = list(reaction_weights.keys())
                weights = list(reaction_weights.values())
                total = sum(weights)
                normalized_weights = [w / total for w in weights]

                social_event.update(
                    {
                        "post_id": f"post_{np.random.randint(100000, 999999)}",
                        "reaction_type": np.random.choice(
                            reactions, p=normalized_weights
                        ),
                        "reaction_strength": np.random.uniform(0.5, 1.0),
                    }
                )
            elif event_type == "comment_posted":
                social_event.update(
                    {
                        "post_id": f"post_{np.random.randint(100000, 999999)}",
                        "comment_text": content.get("text", ""),
                        "comment_length": len(content.get("text", "")),
                        "is_reply": np.random.random() < 0.3,
                    }
                )
            elif event_type == "content_shared":
                social_event.update(
                    {
                        "original_post_id": f"post_{np.random.randint(100000, 999999)}",
                        "share_comment": content.get("text", "")
                        if np.random.random() < 0.5
                        else "",
                        "share_platform": np.random.choice(
                            ["facebook", "twitter", "linkedin"]
                        ),
                    }
                )
            elif event_type == "user_mentioned":
                social_event.update(
                    {
                        "mentioned_user_id": contact if contact else "unknown",
                        "context_post_id": f"post_{np.random.randint(100000, 999999)}",
                        "mention_context": "post"
                        if np.random.random() < 0.7
                        else "comment",
                    }
                )

            # Add emotion metadata
            social_event.update(
                {
                    "sentiment_cue": current_emotion,
                    "ekman_category": ekman_category,
                    "sentiment_valence": sentiment_valence,
                    "emotion_intensity": np.random.uniform(0.3, 0.9),
                }
            )

            events.append(social_event)

        self.events.extend(events)
        return events

    def get_user_emotion_summary(self, user_id: str) -> Dict:
        """Get summary of user's emotion patterns."""
        if user_id not in self.user_emotion_states:
            return {}

        state = self.user_emotion_states[user_id]
        history = state.get("emotion_history", [])

        if not history:
            recent_emotions = []
        else:
            recent_emotions = [h["emotion"] for h in history[-10:]]

        # Count emotion frequencies
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        return {
            "current_emotion": state.get("current_emotion", "unknown"),
            "emotion_stability": state.get("persona_modifier", 1.0),
            "recent_emotions": recent_emotions,
            "emotion_counts": emotion_counts,
            "total_interactions": len(history),
            "last_emotion_change": state.get("last_emotion_change"),
        }
