"""Propensity models for prediction with price sensitivity modeling."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math


class PropensityModels:
    """Calculate propensity scores (LTV, Churn, Conversion) with price sensitivity."""

    def __init__(self, priors: Optional[Dict] = None):
        """Initialize propensity models with price sensitivity."""
        self.ltv_model = None
        self.churn_model = None
        self.conversion_model = None
        self.priors = priors

        from ecosystem_sim.core.prior_loader import get_prior

        self.persona_mult = get_prior(
            priors, "layer_3_intelligence.propensity.conversion_rate_by_persona", None
        ) or {
            "tech_savvy_millennial": 0.18,
            "budget_conscious_parent": 0.12,
            "business_professional": 0.22,
            "casual_browser": 0.05,
            "privacy_focused_user": 0.03,
        }

        # Price sensitivity parameters from dataset
        self.price_sensitivity_params = self._load_price_sensitivity_params()

        # Willingness-to-pay distributions by persona
        self.wtp_distributions = self._load_wtp_distributions()

        # Price elasticity parameters
        self.elasticity_params = {
            "tech_savvy_millennial": -1.2,  # More elastic
            "budget_conscious_parent": -0.8,  # Less elastic
            "business_professional": -0.5,  # Least elastic (price insensitive)
            "casual_browser": -1.5,  # Very elastic
            "privacy_focused_user": -1.0,  # Moderate elasticity
        }

    def _load_price_sensitivity_params(self) -> Dict:
        """Load price sensitivity parameters from priors."""
        if not self.priors:
            return {}

        from ecosystem_sim.core.prior_loader import get_prior

        return get_prior(self.priors, "layer_2_streams.pricing_model", {})

    def _load_wtp_distributions(self) -> Dict:
        """Load willingness-to-pay distributions."""
        # Default distributions if priors not available
        return {
            "tech_savvy_millennial": {"mean": 85.0, "std": 25.0},
            "budget_conscious_parent": {"mean": 45.0, "std": 15.0},
            "business_professional": {"mean": 120.0, "std": 40.0},
            "casual_browser": {"mean": 30.0, "std": 10.0},
            "privacy_focused_user": {"mean": 65.0, "std": 20.0},
        }

    def calculate_ltv(self, user, events: List[Dict], user_profile: Dict) -> float:
        """Calculate Lifetime Value prediction with price sensitivity.

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
        persona_name = user.persona.name.lower().replace(" ", "_")
        persona_val = self.persona_mult.get(persona_name, 0.15)  # Default 15%

        # Price sensitivity adjustment
        price_sensitivity = self._calculate_price_sensitivity(user, events)

        # Project future value with price sensitivity
        base_ltv = total_spend if total_spend > 0 else 100
        ltv = (
            base_ltv
            * (1 + engagement * 0.5)
            * (persona_val / 0.15)
            * (1 + price_sensitivity * 0.2)
        )

        return max(0, ltv)

    def _calculate_price_sensitivity(self, user, events: List[Dict]) -> float:
        """Calculate user's price sensitivity score."""
        persona_name = user.persona.name.lower().replace(" ", "_")

        # Base sensitivity from persona
        base_sensitivity = abs(self.elasticity_params.get(persona_name, -1.0))

        # Adjust based on purchase history
        purchases = [e for e in events if e.get("event_type") == "purchase"]
        if purchases:
            prices = [e.get("price_usd", 0) for e in purchases]
            avg_price = np.mean(prices)

            # Users who buy expensive items are less price sensitive
            if avg_price > 100:
                base_sensitivity *= 0.7
            elif avg_price < 20:
                base_sensitivity *= 1.3

        # Adjust based on discount responsiveness
        discount_events = [
            e
            for e in events
            if e.get("event_type") == "purchase" and e.get("discount_percent", 0) > 0
        ]
        if discount_events:
            discount_rate = len(discount_events) / max(1, len(purchases))
            base_sensitivity *= 1 + discount_rate * 0.5

        return base_sensitivity

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
            datetime.fromisoformat(e.get("timestamp", "").replace("Z", "+00:00"))
            for e in events
            if e.get("timestamp")
        )
        days_inactive = (datetime.now(last_event_time.tzinfo) - last_event_time).days

        # Frequency (events per day)
        first_event_time = min(
            datetime.fromisoformat(e.get("timestamp", "").replace("Z", "+00:00"))
            for e in events
            if e.get("timestamp")
        )
        account_age_days = (last_event_time - first_event_time).days + 1
        frequency = len(events) / max(1, account_age_days)

        # Price sensitivity impact on churn
        price_sensitivity = self._calculate_price_sensitivity(user, events)

        # Base churn risk with price sensitivity adjustment
        # More price-sensitive users are more likely to churn if prices increase
        churn = max(
            0.1,
            min(
                0.9,
                0.5 * (days_inactive / 30) - 0.3 * frequency + 0.1 * price_sensitivity,
            ),
        )

        # Persona adjustment
        persona_name = user.persona.name.lower().replace(" ", "_")
        if persona_name == "tech_savvy_millennial":
            churn *= 0.7
        elif persona_name == "casual_browser":
            churn *= 1.3
        elif persona_name == "budget_conscious_parent":
            churn *= 1.2  # More likely to churn due to budget constraints

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
            persona_name = user.persona.name.lower().replace(" ", "_")
            base_rate = self.persona_mult.get(persona_name, 0.15)
            return base_rate * 0.5

        # Purchase frequency
        purchases = [e for e in events if e.get("event_type") == "purchase"]
        purchase_rate = len(purchases) / max(1, len(events))

        # Browse/search activity
        browse_events = [
            e
            for e in events
            if e.get("event_type") in ["search_query", "product_view", "add_to_cart"]
        ]
        browse_rate = len(browse_events) / max(1, len(events))

        # Price sensitivity impact
        price_sensitivity = self._calculate_price_sensitivity(user, events)

        # Willingness-to-pay estimation
        wtp_score = self._estimate_willingness_to_pay(user, events)

        # Propensity with price sensitivity
        propensity = (
            0.25 * purchase_rate
            + 0.25 * browse_rate
            + 0.25 * wtp_score
            + 0.25 * (1 - price_sensitivity * 0.3)  # Less sensitive = higher conversion
        )

        return np.clip(propensity, 0, 1)

    def _estimate_willingness_to_pay(self, user, events: List[Dict]) -> float:
        """Estimate user's willingness-to-pay score."""
        persona_name = user.persona.name.lower().replace(" ", "_")

        # Base WTP from persona distribution
        wtp_dist = self.wtp_distributions.get(persona_name, {"mean": 50.0, "std": 15.0})
        base_wtp = wtp_dist["mean"]

        # Adjust based on purchase history
        purchases = [e for e in events if e.get("event_type") == "purchase"]
        if purchases:
            prices = [e.get("price_usd", 0) for e in purchases]
            max_price = max(prices) if prices else 0
            avg_price = np.mean(prices) if prices else 0

            # Users who have paid higher prices have higher WTP
            if max_price > base_wtp:
                adjusted_wtp = base_wtp * (1 + (max_price - base_wtp) / (2 * base_wtp))
            else:
                adjusted_wtp = base_wtp * 0.8  # Conservative if never paid high prices
        else:
            adjusted_wtp = base_wtp * 0.7  # No purchase history

        # Normalize to 0-1 scale (assuming WTP range 0-200)
        normalized_wtp = min(1.0, adjusted_wtp / 200.0)

        return normalized_wtp

    def calculate_price_elasticity(self, user, product_category: str) -> float:
        """Calculate price elasticity for a user and product category.

        Args:
            user: User object
            product_category: Product category

        Returns:
            Price elasticity (negative value, more negative = more elastic)
        """
        persona_name = user.persona.name.lower().replace(" ", "_")
        base_elasticity = self.elasticity_params.get(persona_name, -1.0)

        # Adjust for product category
        category_adjustments = {
            "electronics": 1.2,  # More elastic
            "clothing": 1.0,  # Neutral
            "groceries": 0.7,  # Less elastic (necessities)
            "luxury": 0.5,  # Much less elastic (Veblen goods)
            "services": 0.8,  # Less elastic
        }

        adjustment = category_adjustments.get(product_category.lower(), 1.0)
        return base_elasticity * adjustment

    def estimate_optimal_price(
        self, user, product_category: str, cost_price: float
    ) -> Tuple[float, float]:
        """Estimate optimal price for a user and product.

        Args:
            user: User object
            product_category: Product category
            cost_price: Cost price of product

        Returns:
            Tuple of (optimal_price, expected_conversion_rate)
        """
        # Get elasticity
        elasticity = self.calculate_price_elasticity(user, product_category)

        # Estimate WTP
        wtp_dist = self.wtp_distributions.get(
            user.persona.name.lower().replace(" ", "_"), {"mean": 50.0, "std": 15.0}
        )
        mean_wtp = wtp_dist["mean"]

        # Calculate optimal price using elasticity formula
        # For linear demand: Q = a - b*P, optimal P = (a + b*c) / (2b)
        # Simplified: optimal price = (mean_wtp + cost_price) / 2
        optimal_price = (mean_wtp + cost_price) / 2

        # Ensure minimum margin
        min_margin = cost_price * 1.2  # 20% minimum margin
        optimal_price = max(optimal_price, min_margin)

        # Calculate expected conversion rate at optimal price
        price_ratio = optimal_price / mean_wtp
        if price_ratio <= 0.8:
            conversion_rate = 0.7  # Good deal
        elif price_ratio <= 1.0:
            conversion_rate = 0.5  # Fair price
        elif price_ratio <= 1.2:
            conversion_rate = 0.3  # Premium
        else:
            conversion_rate = 0.1  # Expensive

        # Adjust for elasticity
        if elasticity < -1.5:  # Very elastic
            conversion_rate *= 0.8
        elif elasticity > -0.5:  # Inelastic
            conversion_rate *= 1.2

        return optimal_price, conversion_rate

    def calculate_discount_effectiveness(
        self, user, original_price: float, discount_percent: float
    ) -> float:
        """Calculate effectiveness of discount for a user.

        Args:
            user: User object
            original_price: Original price
            discount_percent: Discount percentage (0-100)

        Returns:
            Lift in conversion probability
        """
        persona_name = user.persona.name.lower().replace(" ", "_")

        # Base discount responsiveness by persona
        responsiveness = {
            "tech_savvy_millennial": 1.2,  # Responds well to discounts
            "budget_conscious_parent": 1.5,  # Very responsive
            "business_professional": 0.7,  # Less responsive
            "casual_browser": 1.3,  # Responsive
            "privacy_focused_user": 0.9,  # Neutral
        }

        base_responsiveness = responsiveness.get(persona_name, 1.0)

        # Calculate discount lift
        discount_factor = discount_percent / 100.0

        # Diminishing returns for large discounts
        if discount_factor > 0.5:
            effective_discount = 0.5 + (discount_factor - 0.5) * 0.5
        else:
            effective_discount = discount_factor

        # Calculate lift
        lift = base_responsiveness * effective_discount * 0.3  # Max 30% lift

        # Adjust for price level (discounts on expensive items have more impact)
        if original_price > 100:
            lift *= 1.2
        elif original_price < 20:
            lift *= 0.8

        return min(lift, 0.5)  # Cap at 50% lift

    def calculate_all(self, stitched_data: Dict) -> Dict:
        """Calculate all propensity scores with price sensitivity.

        Args:
            stitched_data: Stitched event data

        Returns:
            Dictionary of propensity scores
        """
        # This would aggregate events by user_id in a real implementation
        # For now, return placeholder structure
        return {
            "ltv_scores": {},
            "churn_risks": {},
            "conversion_propensities": {},
            "price_sensitivities": {},
            "optimal_prices": {},
        }

    def generate_price_sensitivity_report(
        self, users: List, events_by_user: Dict
    ) -> Dict:
        """Generate price sensitivity analysis report.

        Args:
            users: List of user objects
            events_by_user: Dictionary mapping user_id to events

        Returns:
            Price sensitivity report
        """
        report = {
            "summary": {
                "total_users": len(users),
                "avg_price_sensitivity": 0.0,
                "avg_willingness_to_pay": 0.0,
                "price_elasticity_by_persona": {},
            },
            "persona_analysis": {},
            "recommendations": [],
        }

        # Calculate metrics by persona
        persona_metrics = {}

        for user in users:
            persona_name = user.persona.name.lower().replace(" ", "_")
            if persona_name not in persona_metrics:
                persona_metrics[persona_name] = {
                    "count": 0,
                    "price_sensitivity_sum": 0.0,
                    "wtp_sum": 0.0,
                }

            events = events_by_user.get(user.user_id, [])
            price_sensitivity = self._calculate_price_sensitivity(user, events)
            wtp = self._estimate_willingness_to_pay(user, events)

            persona_metrics[persona_name]["count"] += 1
            persona_metrics[persona_name]["price_sensitivity_sum"] += price_sensitivity
            persona_metrics[persona_name]["wtp_sum"] += (
                wtp * 200
            )  # Convert back to dollar scale

        # Populate report
        total_sensitivity = 0
        total_wtp = 0
        total_users = len(users)

        for persona, metrics in persona_metrics.items():
            count = metrics["count"]
            avg_sensitivity = metrics["price_sensitivity_sum"] / count
            avg_wtp = metrics["wtp_sum"] / count

            report["persona_analysis"][persona] = {
                "user_count": count,
                "avg_price_sensitivity": avg_sensitivity,
                "avg_willingness_to_pay": avg_wtp,
                "price_elasticity": self.elasticity_params.get(persona, -1.0),
                "recommended_discount_range": self._get_recommended_discount(
                    persona, avg_wtp
                ),
            }

            total_sensitivity += avg_sensitivity * count
            total_wtp += avg_wtp * count

        # Calculate overall averages
        if total_users > 0:
            report["summary"]["avg_price_sensitivity"] = total_sensitivity / total_users
            report["summary"]["avg_willingness_to_pay"] = total_wtp / total_users

        # Add persona-specific elasticity
        for persona in persona_metrics.keys():
            report["summary"]["price_elasticity_by_persona"][persona] = (
                self.elasticity_params.get(persona, -1.0)
            )

        # Generate recommendations
        report["recommendations"] = self._generate_price_recommendations(
            persona_metrics
        )

        return report

    def _get_recommended_discount(self, persona: str, avg_wtp: float) -> Dict:
        """Get recommended discount range for a persona."""
        recommendations = {
            "tech_savvy_millennial": {"min": 10, "max": 25},
            "budget_conscious_parent": {"min": 15, "max": 30},
            "business_professional": {"min": 5, "max": 15},
            "casual_browser": {"min": 20, "max": 40},
            "privacy_focused_user": {"min": 10, "max": 20},
        }

        # Adjust based on WTP
        base_rec = recommendations.get(persona, {"min": 10, "max": 20})
        if avg_wtp < 40:
            # Lower WTP, need higher discounts
            return {"min": base_rec["min"] + 5, "max": base_rec["max"] + 10}
        elif avg_wtp > 100:
            # Higher WTP, can use lower discounts
            return {
                "min": max(5, base_rec["min"] - 5),
                "max": max(10, base_rec["max"] - 10),
            }
        else:
            return base_rec

    def _generate_price_recommendations(self, persona_metrics: Dict) -> List[str]:
        """Generate price optimization recommendations."""
        recommendations = []

        # Check for high price sensitivity segments
        for persona, metrics in persona_metrics.items():
            avg_sensitivity = metrics["price_sensitivity_sum"] / metrics["count"]

            if avg_sensitivity > 1.2:
                recommendations.append(
                    f"High price sensitivity detected for {persona.replace('_', ' ').title()} "
                    f"segment. Consider implementing targeted discounts or bundling strategies."
                )
            elif avg_sensitivity < 0.7:
                recommendations.append(
                    f"Low price sensitivity detected for {persona.replace('_', ' ').title()} "
                    f"segment. Opportunity for premium pricing or value-added services."
                )

        # Check WTP variations
        wtp_values = []
        for persona, metrics in persona_metrics.items():
            avg_wtp = metrics["wtp_sum"] / metrics["count"]
            wtp_values.append((persona, avg_wtp))

        if wtp_values:
            max_wtp_persona, max_wtp = max(wtp_values, key=lambda x: x[1])
            min_wtp_persona, min_wtp = min(wtp_values, key=lambda x: x[1])

            if max_wtp > min_wtp * 1.5:
                recommendations.append(
                    f"Significant WTP variation detected: {max_wtp_persona.replace('_', ' ').title()} "
                    f"({max_wtp:.0f}) vs {min_wtp_persona.replace('_', ' ').title()} ({min_wtp:.0f}). "
                    f"Consider persona-based pricing tiers."
                )

        # Add general recommendations
        recommendations.extend(
            [
                "Implement dynamic pricing based on user persona and purchase history.",
                "Use A/B testing to validate price elasticity estimates.",
                "Monitor competitor pricing for key product categories.",
                "Consider time-based pricing for seasonal or time-sensitive products.",
            ]
        )

        return recommendations
