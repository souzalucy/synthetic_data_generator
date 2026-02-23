"""Search stream event generation."""

import numpy as np
from typing import List, Dict
from .base_stream import BaseStream


class SearchStream(BaseStream):
    """Generates search-related events."""
    
    def __init__(self, config: Dict, state_manager, taxonomy: Dict):
        """Initialize search stream."""
        super().__init__(config, state_manager, taxonomy, "SEARCH_STREAM")
        self.categories = list(taxonomy["categories"].keys())
    
    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate search events for a user.
        
        Args:
            user: User object
            timestamp: ISO format timestamp
            probabilities: Action probabilities
            
        Returns:
            List of events
        """
        events = []
        
        # Find search actions in probabilities
        search_actions = [k for k in probabilities.keys() if k.startswith("search_query_")]
        
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
                
                # Generate query
                query = self._generate_query(category, user)
                
                # Create search_query event
                event = self.create_base_event(user.user_id, device_id, timestamp, "search_query")
                event.update({
                    "query": query,
                    "category": category,
                    "query_length": len(query.split()),
                })
                events.append(event)
                
                # Probabilistically generate follow-up events
                if self.should_generate_event(0.7):  # 70% chance of SERP view
                    serp_event = self.create_base_event(user.user_id, device_id, timestamp, "serp_view")
                    serp_event.update({
                        "query": query,
                        "category": category,
                        "result_count": np.random.randint(5, 100),
                    })
                    events.append(serp_event)
                    
                    # 30% chance of organic click
                    if self.should_generate_event(0.3):
                        click_event = self.create_base_event(user.user_id, device_id, timestamp, "organic_click")
                        click_event.update({
                            "query": query,
                            "category": category,
                            "result_position": np.random.randint(1, 11),
                        })
                        events.append(click_event)
                
                # Ad impression and click
                if self.should_generate_event(0.4):  # 40% see ads
                    ad_event = self.create_base_event(user.user_id, device_id, timestamp, "ad_impression")
                    ad_event.update({
                        "query": query,
                        "category": category,
                        "ad_id": f"ad_{np.random.randint(1000, 9999)}",
                    })
                    events.append(ad_event)
                    
                    # Ad click if persona has propensity
                    if self.should_generate_event(user.persona.ad_click_propensity):
                        ad_click = self.create_base_event(user.user_id, device_id, timestamp, "ad_click")
                        ad_click.update({
                            "query": query,
                            "category": category,
                            "ad_id": ad_event["ad_id"],
                        })
                        events.append(ad_click)
        
        self.events.extend(events)
        return events
    
    def _generate_query(self, category: str, user) -> str:
        """Generate realistic search query.
        
        Args:
            category: Product category
            user: User object
            
        Returns:
            Search query string
        """
        category_data = self.taxonomy["categories"][category]
        keywords = category_data["keywords"]
        
        # Sample 1-4 keywords to form query
        n_words = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        query_words = np.random.choice(keywords, size=min(n_words, len(keywords)), replace=False)
        
        query = " ".join(query_words)
        
        # Add modifiers based on persona
        if user.persona.tech_savviness.value > 0.7:
            modifiers = ["best", "latest", "2024", "top rated"]
            if np.random.random() < 0.3:
                query = np.random.choice(modifiers) + " " + query
        
        return query
