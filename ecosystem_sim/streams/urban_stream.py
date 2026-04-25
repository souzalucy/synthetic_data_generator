"""Urban mobility and geospatial interaction stream."""

import numpy as np
from typing import List, Dict, Optional
from .base_stream import BaseStream


class UrbanStream(BaseStream):
    """Generates urban mobility and building interaction events."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        super().__init__(config, state_manager, taxonomy, "URBAN_STREAM", priors)

        from ecosystem_sim.core.prior_loader import get_prior

        # User state tracking for sector transitions
        self.user_sectors = {}

        # Load urban density patterns
        self.building_density_patterns = get_prior(
            priors, "layer_1_persona.urban_density_patterns", {}
        )

        self.sector_mobility_rates = get_prior(
            priors,
            "layer_2_streams.urban.sector_transition_matrix",
            self._create_default_sector_transition_matrix(),
        )

        self.poi_visit_rates = get_prior(
            priors,
            "layer_2_streams.urban.poi_visit_pattern",
            {
                "restaurant": 0.25,
                "retail": 0.2,
                "entertainment": 0.15,
                "transport": 0.4,
            },
        )

        self.building_interaction_rates = get_prior(
            priors,
            "layer_2_streams.urban.building_interaction_rates",
            {
                "residential_entry": 2.1,
                "commercial_entry": 1.8,
                "industrial_entry": 0.5,
                "mixed_use_entry": 0.7,
            },
        )

        self.mobility_modes = get_prior(
            priors,
            "layer_2_streams.urban.mobility_modes",
            {"walk": 0.2, "bike": 0.1, "car": 0.5, "transit": 0.2},
        )

    def _create_default_sector_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Create default sector transition probability matrix."""
        sectors = [
            "residential",
            "commercial",
            "industrial",
            "mixed",
            "educational",
            "transport",
        ]

        # Base transition probabilities
        matrix = {}
        for from_sector in sectors:
            matrix[from_sector] = {}
            total = 0
            for to_sector in sectors:
                if from_sector == to_sector:
                    prob = 0.4  # Stay in same sector
                elif (from_sector == "residential" and to_sector == "commercial") or (
                    from_sector == "commercial" and to_sector == "residential"
                ):
                    prob = 0.25  # Home-work commute
                else:
                    prob = 0.05  # Other transitions
                matrix[from_sector][to_sector] = prob
                total += prob

            # Normalize
            for to_sector in sectors:
                matrix[from_sector][to_sector] /= total

        return matrix

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate urban mobility events."""
        events = []

        # Building entry/exit events
        if self.should_generate_event(probabilities.get("mobility", 0.3)):
            events.append(self._create_building_interaction_event(user, timestamp))

        # Sector transition events
        if self.should_generate_event(probabilities.get("sector_transition", 0.2)):
            events.append(self._create_sector_transition_event(user, timestamp))

        # POI checkin events
        if self.should_generate_event(probabilities.get("poi_checkin", 0.15)):
            events.append(self._create_poi_checkin_event(user, timestamp))

        self.events.extend(events)
        return events

    def _create_building_interaction_event(self, user, timestamp: str) -> Dict:
        """Create building entry/exit event."""
        device_id = user.devices[0].device_id if user.devices else "unknown"
        base_event = self.create_base_event(
            user.user_id, device_id, timestamp, "BUILDING_INTERACTION", user=user
        )

        # Choose building type based on urban density patterns
        building_types = list(self.building_density_patterns.keys())
        building_probs = list(self.building_density_patterns.values())

        if not building_types:
            building_types = ["residential", "commercial", "industrial", "mixed_use"]
            building_probs = [0.4, 0.3, 0.2, 0.1]

        building_type = np.random.choice(building_types, p=building_probs)

        # Determine height category
        if building_type == "residential":
            height_categories = ["low_density", "medium_density", "high_density"]
            height_probs = [0.5, 0.3, 0.2]
        elif building_type == "commercial":
            height_categories = ["medium_density", "high_density"]
            height_probs = [0.6, 0.4]
        else:
            height_categories = ["low_density", "medium_density"]
            height_probs = [0.7, 0.3]

        height_category = np.random.choice(height_categories, p=height_probs)

        # Determine floor
        if height_category == "high_density":
            floor = int(np.random.randint(1, 20))
        elif height_category == "medium_density":
            floor = int(np.random.randint(1, 8))
        else:  # low_density
            floor = 1

        return {
            **base_event,
            "action": np.random.choice(["enter", "exit"]),
            "building_type": building_type,
            "density_category": height_category,
            "floor": floor,
            "purpose": np.random.choice(
                ["work", "shopping", "leisure", "transit", "home"],
                p=[0.3, 0.2, 0.15, 0.15, 0.2],
            ),
            "duration_minutes": int(np.random.exponential(60))
            if building_type == "commercial"
            else int(np.random.exponential(120)),
        }

    def _create_sector_transition_event(self, user, timestamp: str) -> Dict:
        """Create sector transition event."""
        device_id = user.devices[0].device_id if user.devices else "unknown"
        base_event = self.create_base_event(
            user.user_id, device_id, timestamp, "SECTOR_TRANSITION", user=user
        )

        sectors = list(self.taxonomy.get("sectors", {}).keys())
        if not sectors:
            sectors = [
                "residential",
                "commercial",
                "industrial",
                "mixed",
                "educational",
                "transport",
            ]

        # Get current sector from user state or random
        if user.user_id not in self.user_sectors:
            self.user_sectors[user.user_id] = {
                "current_sector": np.random.choice(sectors)
            }

        user_state = self.user_sectors[user.user_id]
        from_sector = user_state.get("current_sector", np.random.choice(sectors))

        # Choose destination sector based on transition matrix
        if from_sector in self.sector_mobility_rates:
            to_sector_probs = self.sector_mobility_rates[from_sector]
            to_sectors = list(to_sector_probs.keys())
            to_probs = list(to_sector_probs.values())
            to_sector = np.random.choice(to_sectors, p=to_probs)
        else:
            to_sector = np.random.choice([s for s in sectors if s != from_sector])

        # Update user state
        user_state["current_sector"] = to_sector

        # Choose transport mode
        transport_modes = list(self.mobility_modes.keys())
        transport_probs = list(self.mobility_modes.values())
        transport_mode = np.random.choice(transport_modes, p=transport_probs)

        # Calculate distance based on sector types
        if from_sector == "residential" and to_sector == "commercial":
            distance = float(np.random.lognormal(2.0, 0.5))  # 5-15 km commute
        else:
            distance = float(np.random.lognormal(1.5, 0.8))  # 2-10 km

        return {
            **base_event,
            "action": "sector_transition",
            "from_sector": from_sector,
            "to_sector": to_sector,
            "distance_km": distance,
            "transport_mode": transport_mode,
            "travel_time_minutes": int(
                distance * np.random.uniform(2, 10)
            ),  # 2-10 min per km
            "carbon_footprint_kg": distance
            * {"walk": 0, "bike": 0, "car": 0.12, "transit": 0.05}[transport_mode],
        }

    def _create_poi_checkin_event(self, user, timestamp: str) -> Dict:
        """Create POI checkin event."""
        device_id = user.devices[0].device_id if user.devices else "unknown"
        base_event = self.create_base_event(
            user.user_id, device_id, timestamp, "POI_CHECKIN", user=user
        )

        poi_types = list(self.poi_visit_rates.keys())
        poi_probs = list(self.poi_visit_rates.values())

        if not poi_types:
            poi_types = ["restaurant", "retail", "entertainment", "transport"]
            poi_probs = [0.25, 0.2, 0.15, 0.4]

        poi_type = np.random.choice(poi_types, p=poi_probs)

        # Generate checkin details
        if poi_type == "restaurant":
            cuisine = np.random.choice(
                ["italian", "chinese", "mexican", "indian", "american"]
            )
            rating = np.random.uniform(3.5, 5.0)
            price_level = np.random.choice(["$", "$$", "$$$"], p=[0.5, 0.3, 0.2])
        elif poi_type == "retail":
            category = np.random.choice(
                ["clothing", "electronics", "grocery", "home_goods"]
            )
            rating = np.random.uniform(3.0, 5.0)
            price_level = np.random.choice(["$", "$$", "$$$"], p=[0.6, 0.3, 0.1])
        elif poi_type == "entertainment":
            category = np.random.choice(["cinema", "theater", "museum", "park"])
            rating = np.random.uniform(3.5, 5.0)
            price_level = np.random.choice(["$", "$$", "$$$"], p=[0.7, 0.2, 0.1])
        else:  # transport
            category = np.random.choice(["station", "airport", "bus_terminal", "port"])
            rating = np.random.uniform(2.5, 4.5)
            price_level = "$"

        return {
            **base_event,
            "poi_type": poi_type,
            "category": category if "category" in locals() else "general",
            "cuisine": cuisine if "cuisine" in locals() else None,
            "rating": round(rating, 1),
            "price_level": price_level,
            "dwell_time_minutes": int(np.random.exponential(45)),
            "social_context": np.random.choice(
                ["alone", "with_friends", "with_family", "with_colleagues"],
                p=[0.3, 0.3, 0.2, 0.2],
            ),
        }

    def should_generate_event(self, probability: float) -> bool:
        """Determine if an event should be generated based on probability."""
        return np.random.random() < probability
