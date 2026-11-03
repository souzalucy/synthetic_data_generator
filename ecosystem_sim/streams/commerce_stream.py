"""Commerce stream event generation."""

import numpy as np
from typing import List, Dict, Optional
from .base_stream import BaseStream


class CommerceStream(BaseStream):
    """Generates e-commerce related events."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        """Initialize commerce stream with Instacart patterns."""
        super().__init__(config, state_manager, taxonomy, "COMMERCE_STREAM", priors)
        self.categories = list(taxonomy["categories"].keys())

        from ecosystem_sim.core.prior_loader import get_prior

        # Load Instacart-specific patterns from priors
        self.instacart_reorder_rate = get_prior(
            priors, "layer_2_streams.commerce.instacart_reorder_rate", 0.59
        )

        self.day_of_week_pattern = get_prior(
            priors,
            "layer_2_streams.commerce.day_of_week_pattern",
            [0.12, 0.13, 0.14, 0.15, 0.18, 0.16, 0.12],  # Sunday-Saturday
        )

        self.hour_of_day_pattern = get_prior(
            priors,
            "layer_2_streams.commerce.hour_of_day_pattern",
            [
                0.01,
                0.00,
                0.00,
                0.00,
                0.01,
                0.02,
                0.04,
                0.06,
                0.08,
                0.09,
                0.10,
                0.11,
                0.12,
                0.10,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.02,
                0.01,
                0.01,
            ],
        )

        self.avg_days_between_orders = get_prior(
            priors, "layer_2_streams.commerce.avg_days_between_orders", 7.2
        )

        # Product reorder probability distribution
        self.reorder_prob_dist = get_prior(
            priors,
            "layer_2_streams.commerce.reorder_probability_distribution",
            {"shape": 2.5, "scale": 7.5},  # Beta distribution parameters
        )

        # Standard rates
        self.cart_to_purchase_rate = get_prior(
            priors, "layer_2_streams.commerce.cart_to_purchase_rate", 0.7
        )
        self.review_rate = get_prior(
            priors, "layer_2_streams.commerce.review_rate", 0.1
        )

        basket_dist = get_prior(
            priors, "layer_2_streams.commerce.basket_size_distribution", None
        )
        self.basket_mean = (
            basket_dist["mean"] if basket_dist and "mean" in basket_dist else 8.3
        )  # Instacart average
        self.basket_std = (
            basket_dist["std"] if basket_dist and "std" in basket_dist else 4.0
        )

        # Track user purchase history for reorder patterns
        self.user_purchase_history = {}  # user_id -> list of purchase events

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate commerce events for a user.

        Args:
            user: User object
            timestamp: ISO format timestamp
            probabilities: Action probabilities

        Returns:
            List of events
        """
        events = []

        # Find commerce actions
        commerce_actions = [
            k for k in probabilities.keys() if k.startswith("commerce_browse_")
        ]

        for action in commerce_actions:
            prob = probabilities[action]

            if self.should_generate_event(prob):
                cat_idx = int(action.split("_")[-1])
                if cat_idx < len(self.categories):
                    category = self.categories[cat_idx]
                else:
                    category = np.random.choice(self.categories)

                device_id = self.state_manager.get_device_for_action(user, "commerce")

                # Shopping funnel
                n_views = max(
                    1, int(np.random.normal(self.basket_mean, self.basket_std))
                )
                n_views = min(n_views, 10)

                # Product views
                for _ in range(n_views):
                    product = np.random.choice(
                        self.taxonomy["categories"][category]["products"]
                    )
                    price = np.random.uniform(
                        *self.taxonomy["categories"][category]["price_range"]
                    )

                    view_event = self.create_base_event(
                        user.user_id, device_id, timestamp, "product_view", user=user
                    )
                    view_event.update(
                        {
                            "product_id": f"prod_{hash(product) % 100000}",
                            "product_name": product,
                            "category": category,
                            "price_usd": round(price, 2),
                        }
                    )
                    events.append(view_event)

                # Add to cart (70% probability)
                if self.should_generate_event(self.cart_to_purchase_rate):
                    cart_event = self.create_base_event(
                        user.user_id, device_id, timestamp, "add_to_cart", user=user
                    )
                    cart_event.update(
                        {
                            "product_id": view_event["product_id"],
                            "category": category,
                            "quantity": 1,
                            "price_usd": view_event["price_usd"],
                        }
                    )
                    events.append(cart_event)

                    # Purchase (30% * impulse_buying)
                    purchase_prob = 0.3 * user.persona.impulse_buying
                    if self.should_generate_event(purchase_prob):
                        purchase_event = self.create_base_event(
                            user.user_id, device_id, timestamp, "purchase", user=user
                        )
                        purchase_event.update(
                            {
                                "product_id": cart_event["product_id"],
                                "product_name": view_event["product_name"],
                                "category": category,
                                "price_usd": cart_event["price_usd"],
                                "quantity": 1,
                                "payment_method": np.random.choice(
                                    ["credit_card", "paypal", "debit_card"]
                                ),
                                "shipping_method": np.random.choice(
                                    ["standard", "express", "overnight"]
                                ),
                            }
                        )
                        events.append(purchase_event)

                        # Track purchase in user history
                        user.purchase_history.append(
                            {
                                "product_id": purchase_event["product_id"],
                                "category": category,
                                "price_usd": purchase_event["price_usd"],
                                "timestamp": timestamp,
                            }
                        )

                        # Small chance of review
                        if self.should_generate_event(self.review_rate):
                            review_event = self.create_base_event(
                                user.user_id,
                                device_id,
                                timestamp,
                                "product_review",
                                user=user,
                            )
                            review_event.update(
                                {
                                    "product_id": purchase_event["product_id"],
                                    "rating": np.random.randint(1, 6),
                                    "review_text": f"Great product!"
                                    if np.random.random() > 0.3
                                    else "Not as described",
                                }
                            )
                            events.append(review_event)
                    else:
                        # Cart abandonment
                        abandon_event = self.create_base_event(
                            user.user_id,
                            device_id,
                            timestamp,
                            "cart_abandonment",
                            user=user,
                        )
                        abandon_event.update(
                            {
                                "product_id": cart_event["product_id"],
                                "category": category,
                                "value_usd": cart_event["price_usd"],
                            }
                        )
                        events.append(abandon_event)

        return events
