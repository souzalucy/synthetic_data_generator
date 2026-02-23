"""Commerce stream event generation."""

import numpy as np
from typing import List, Dict
from .base_stream import BaseStream


class CommerceStream(BaseStream):
    """Generates e-commerce related events."""
    
    def __init__(self, config: Dict, state_manager, taxonomy: Dict):
        """Initialize commerce stream."""
        super().__init__(config, state_manager, taxonomy, "COMMERCE_STREAM")
        self.categories = list(taxonomy["categories"].keys())
    
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
        commerce_actions = [k for k in probabilities.keys() if k.startswith("commerce_browse_")]
        
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
                n_views = np.random.randint(1, 5)
                
                # Product views
                for _ in range(n_views):
                    product = np.random.choice(self.taxonomy["categories"][category]["products"])
                    price = np.random.uniform(*self.taxonomy["categories"][category]["price_range"])
                    
                    view_event = self.create_base_event(user.user_id, device_id, timestamp, "product_view")
                    view_event.update({
                        "product_id": f"prod_{hash(product) % 100000}",
                        "product_name": product,
                        "category": category,
                        "price_usd": round(price, 2),
                    })
                    events.append(view_event)
                
                # Add to cart (70% probability)
                if self.should_generate_event(0.7):
                    cart_event = self.create_base_event(user.user_id, device_id, timestamp, "add_to_cart")
                    cart_event.update({
                        "product_id": view_event["product_id"],
                        "category": category,
                        "quantity": 1,
                        "price_usd": view_event["price_usd"],
                    })
                    events.append(cart_event)
                    
                    # Purchase (30% * impulse_buying)
                    purchase_prob = 0.3 * user.persona.impulse_buying
                    if self.should_generate_event(purchase_prob):
                        purchase_event = self.create_base_event(user.user_id, device_id, timestamp, "purchase")
                        purchase_event.update({
                            "product_id": cart_event["product_id"],
                            "product_name": view_event["product_name"],
                            "category": category,
                            "price_usd": cart_event["price_usd"],
                            "quantity": 1,
                            "payment_method": np.random.choice(["credit_card", "paypal", "debit_card"]),
                            "shipping_method": np.random.choice(["standard", "express", "overnight"]),
                        })
                        events.append(purchase_event)
                        
                        # Track purchase in user history
                        user.purchase_history.append({
                            "product_id": purchase_event["product_id"],
                            "category": category,
                            "price_usd": purchase_event["price_usd"],
                            "timestamp": timestamp,
                        })
                        
                        # Small chance of review
                        if self.should_generate_event(0.1):
                            review_event = self.create_base_event(user.user_id, device_id, timestamp, "product_review")
                            review_event.update({
                                "product_id": purchase_event["product_id"],
                                "rating": np.random.randint(1, 6),
                                "review_text": f"Great product!" if np.random.random() > 0.3 else "Not as described",
                            })
                            events.append(review_event)
                    else:
                        # Cart abandonment
                        abandon_event = self.create_base_event(user.user_id, device_id, timestamp, "cart_abandonment")
                        abandon_event.update({
                            "product_id": cart_event["product_id"],
                            "category": category,
                            "value_usd": cart_event["price_usd"],
                        })
                        events.append(abandon_event)
        
        self.events.extend(events)
        return events
