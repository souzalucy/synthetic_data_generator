"""Geographic stream for location-based events with GeoLife GPS trajectory patterns."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .base_stream import BaseStream
from datetime import datetime, timedelta
import random
import math


class GeoStream(BaseStream):
    """Generates location-based events with GeoLife GPS trajectory patterns."""

    def __init__(
        self, config: Dict, state_manager, taxonomy: Dict, priors: Optional[Dict] = None
    ):
        """Initialize geo stream with GeoLife patterns."""
        super().__init__(config, state_manager, taxonomy, "GEO_STREAM", priors)
        self.sectors = list(taxonomy["sectors"].keys())
        self.user_locations = {}
        self.user_trajectories = {}
        self.user_movement_state = {}

        # Load GeoLife priors
        from ecosystem_sim.core.prior_loader import get_prior

        hw_dist = get_prior(priors, "layer_2_streams.geo.home_work_distance_km", None)
        if hw_dist and "fit_params" in hw_dist:
            self._hw_mu = hw_dist["fit_params"].get("mu", 1.8)
            self._hw_sigma = hw_dist["fit_params"].get("sigma", 0.8)
        else:
            self._hw_mu = 1.8
            self._hw_sigma = 0.8

        # GeoLife trajectory patterns (based on dataset analysis)
        self.trajectory_length_dist = {
            "mean_km": 4.2,
            "std_km": 3.8,
            "min_km": 0.1,
            "max_km": 25.0,
        }

        self.dwell_time_dist = {
            "home": {"mean_min": 420, "std_min": 180},  # 7 hours ± 3
            "work": {"mean_min": 480, "std_min": 120},  # 8 hours ± 2
            "leisure": {"mean_min": 45, "std_min": 30},  # 45 min ± 30
            "other": {"mean_min": 45, "std_min": 30},
        }

        self.movement_speeds = {
            "walking": {"mean_kmh": 4.2, "std_kmh": 1.1},
            "cycling": {"mean_kmh": 15.3, "std_kmh": 3.2},
            "driving": {"mean_kmh": 35.7, "std_kmh": 12.4},
        }

        # Location clustering (based on GeoLife user patterns)
        self.location_clusters = self._generate_location_clusters()

    def _generate_location_clusters(self) -> List[Tuple[float, float, str]]:
        """Generate realistic location clusters based on urban patterns."""
        clusters = []
        # Home clusters (residential areas)
        for _ in range(5):
            lat = np.random.uniform(40.6, 40.9)
            lon = np.random.uniform(-74.2, -73.8)
            clusters.append((lat, lon, "home_cluster"))

        # Work clusters (commercial districts)
        for _ in range(3):
            lat = np.random.uniform(40.7, 40.8)
            lon = np.random.uniform(-74.0, -73.9)
            clusters.append((lat, lon, "work_cluster"))

        # Leisure clusters (parks, malls, restaurants)
        for _ in range(4):
            lat = np.random.uniform(40.65, 40.85)
            lon = np.random.uniform(-74.1, -73.95)
            clusters.append((lat, lon, "leisure_cluster"))

        return clusters

    def _initialize_user_locations(self, user_id: str) -> Dict:
        """Initialize user's home, work, and frequent locations."""
        # Select home cluster
        home_clusters = [c for c in self.location_clusters if c[2] == "home_cluster"]
        home_lat, home_lon, _ = random.choice(home_clusters)

        # Add small random offset within cluster
        home_lat += np.random.normal(0, 0.005)
        home_lon += np.random.normal(0, 0.005)

        # Generate work location based on home-work distance distribution
        hw_km = np.random.lognormal(self._hw_mu, self._hw_sigma)
        hw_km = np.clip(hw_km, 0.5, 50.0)
        offset_deg = hw_km / 111.0
        angle = np.random.uniform(0, 2 * np.pi)

        work_lat = home_lat + offset_deg * math.cos(angle)
        work_lon = home_lon + offset_deg * math.sin(angle)

        # Generate 2-3 frequent leisure locations
        leisure_locations = []
        leisure_clusters = [
            c for c in self.location_clusters if c[2] == "leisure_cluster"
        ]
        for _ in range(np.random.randint(2, 4)):
            cluster_lat, cluster_lon, _ = random.choice(leisure_clusters)
            lat = cluster_lat + np.random.normal(0, 0.003)
            lon = cluster_lon + np.random.normal(0, 0.003)
            leisure_locations.append((lat, lon, f"leisure_{_}"))

        return {
            "home": (home_lat, home_lon),
            "work": (work_lat, work_lon),
            "leisure": leisure_locations,
            "last_location": "home",
            "last_update": None,
            "dwell_start": None,
            "current_trajectory": None,
        }

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance in kilometers."""
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371 * c

    def _get_movement_type(self, distance_km: float, hour: int) -> str:
        """Determine movement type based on distance and time."""
        if distance_km < 1.0:
            return "walking"
        elif distance_km < 5.0:
            # Cycling more common during daytime
            if 6 <= hour < 20:
                return "cycling" if np.random.random() < 0.3 else "walking"
            else:
                return "walking"
        else:
            return "driving"

    def _simulate_trajectory(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        movement_type: str,
    ) -> List[Tuple[float, float]]:
        """Simulate a GPS trajectory between two points."""
        # Number of points based on movement type
        distance_km = self._calculate_distance(start_lat, start_lon, end_lat, end_lon)

        if movement_type == "walking":
            n_points = max(3, int(distance_km * 4))
        elif movement_type == "cycling":
            n_points = max(3, int(distance_km * 2))
        else:  # driving
            n_points = max(3, int(distance_km))

        # Generate intermediate points with some randomness
        trajectory = []
        for i in range(n_points + 1):
            t = i / n_points
            # Linear interpolation with some noise
            lat = start_lat + (end_lat - start_lat) * t
            lon = start_lon + (end_lon - start_lon) * t

            # Add realistic GPS noise (5-15 meters)
            lat_noise = np.random.normal(0, 0.0001)
            lon_noise = np.random.normal(0, 0.0001)

            trajectory.append((lat + lat_noise, lon + lon_noise))

        return trajectory

    def _get_next_location(
        self, current_loc: str, hour: int, weekday: int, locations: Dict
    ) -> Tuple[str, float, float]:
        """Determine next location based on time, day, and user patterns."""
        if weekday < 5:  # Weekday
            if current_loc == "home" and 7 <= hour < 9:
                return "work", locations["work"][0], locations["work"][1]
            elif current_loc == "work" and 17 <= hour < 19:
                return "home", locations["home"][0], locations["home"][1]
            elif current_loc in ["home", "work"] and 12 <= hour < 14:
                # Lunch break - might go to leisure location
                if locations["leisure"] and np.random.random() < 0.4:
                    leisure = random.choice(locations["leisure"])
                    return "leisure", leisure[0], leisure[1]

        # Random movement to leisure or return home
        if locations["leisure"] and np.random.random() < 0.3:
            leisure = random.choice(locations["leisure"])
            return "leisure", leisure[0], leisure[1]
        else:
            return "home", locations["home"][0], locations["home"][1]

    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate geo events with GeoLife trajectory patterns.

        Args:
            user: User object
            timestamp: ISO format timestamp
            probabilities: Action probabilities

        Returns:
            List of events
        """
        events = []

        # Geo events only on mobile devices
        mobile_devices = [d for d in user.devices if d.device_type == "mobile"]
        if not mobile_devices:
            return events

        if "geo_update" not in probabilities or probabilities["geo_update"] < 0.01:
            return events

        device_id = np.random.choice([d.device_id for d in mobile_devices])

        # Initialize user locations if needed
        if user.user_id not in self.user_locations:
            self.user_locations[user.user_id] = self._initialize_user_locations(
                user.user_id
            )
            self.user_movement_state[user.user_id] = {
                "current_location": "home",
                "dwell_start": timestamp,
                "in_motion": False,
                "trajectory": None,
                "trajectory_index": 0,
                "destination": None,
            }

        locations = self.user_locations[user.user_id]
        movement_state = self.user_movement_state[user.user_id]

        # Parse timestamp
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        hour = ts.hour
        weekday = ts.weekday()

        # Determine if we should generate a GPS update
        if not self.should_generate_event(probabilities["geo_update"]):
            return events

        # Handle movement state
        if movement_state["in_motion"]:
            # User is moving along a trajectory
            if movement_state["trajectory"] and movement_state[
                "trajectory_index"
            ] < len(movement_state["trajectory"]):
                # Get next point in trajectory
                lat, lon = movement_state["trajectory"][
                    movement_state["trajectory_index"]
                ]
                movement_state["trajectory_index"] += 1

                # Check if reached destination
                if movement_state["trajectory_index"] >= len(
                    movement_state["trajectory"]
                ):
                    movement_state["in_motion"] = False
                    movement_state["current_location"] = movement_state["destination"]
                    movement_state["dwell_start"] = timestamp
                    movement_state["trajectory"] = None
                    movement_state["trajectory_index"] = 0
                    movement_state["destination"] = None

                location_type = "in_transit"
            else:
                # Shouldn't happen, but reset if trajectory issues
                movement_state["in_motion"] = False
                lat, lon = locations["home"]
                location_type = "home"
        else:
            # User is stationary at a location
            current_loc = movement_state["current_location"]
            if current_loc == "home":
                lat, lon = locations["home"]
                location_type = "home"
            elif current_loc == "work":
                lat, lon = locations["work"]
                location_type = "work"
            elif current_loc == "leisure":
                if locations["leisure"]:
                    # Use the first leisure location as default
                    lat, lon = locations["leisure"][0][0], locations["leisure"][0][1]
                    location_type = "leisure"
                else:
                    lat, lon = locations["home"]
                    location_type = "home"
            else:
                lat, lon = locations["home"]
                location_type = "home"

            # Check if it's time to move
            dwell_start = movement_state["dwell_start"]
            if dwell_start:
                dwell_dt = datetime.fromisoformat(dwell_start.replace("Z", "+00:00"))
                dwell_minutes = (ts - dwell_dt).total_seconds() / 60

                # Check if dwell time exceeded
                dwell_dist = self.dwell_time_dist.get(
                    location_type, self.dwell_time_dist["other"]
                )
                expected_dwell = np.random.normal(
                    dwell_dist["mean_min"], dwell_dist["std_min"]
                )
                expected_dwell = max(5, expected_dwell)

                if dwell_minutes > expected_dwell and np.random.random() < 0.3:
                    # Start moving to new location
                    dest_type, dest_lat, dest_lon = self._get_next_location(
                        location_type, hour, weekday, locations
                    )

                    distance_km = self._calculate_distance(lat, lon, dest_lat, dest_lon)
                    movement_type = self._get_movement_type(distance_km, hour)

                    # Simulate trajectory
                    trajectory = self._simulate_trajectory(
                        lat, lon, dest_lat, dest_lon, movement_type
                    )

                    movement_state["in_motion"] = True
                    movement_state["trajectory"] = trajectory
                    movement_state["trajectory_index"] = 0
                    movement_state["destination"] = dest_type

                    # Use first point of trajectory
                    if trajectory:
                        lat, lon = trajectory[0]
                        location_type = "in_transit"

        # Create GPS event
        gps_event = self.create_base_event(
            user.user_id, device_id, timestamp, "gps_update", user=user
        )
        gps_event.update(
            {
                "latitude": lat,
                "longitude": lon,
                "accuracy_meters": np.random.uniform(5, 25),
                "location_type": location_type,
                "home_sector": user.home_sector,
                "speed_kmh": np.random.normal(
                    self.movement_speeds.get("walking", {"mean_kmh": 4.2})["mean_kmh"],
                    self.movement_speeds.get("walking", {"std_kmh": 1.1})["std_kmh"],
                )
                if location_type == "in_transit"
                else 0.0,
                "bearing_degrees": np.random.uniform(0, 360)
                if location_type == "in_transit"
                else 0.0,
            }
        )
        events.append(gps_event)

        # Probabilistically add WiFi event when stationary
        if location_type != "in_transit" and self.should_generate_event(0.4):
            wifi_event = self.create_base_event(
                user.user_id, device_id, timestamp, "wifi_connection", user=user
            )
            ssid = f"WiFi_{location_type}_{np.random.randint(1, 100)}"
            wifi_event.update(
                {
                    "ssid": ssid,
                    "location_type": location_type,
                    "signal_strength_dbm": np.random.uniform(-70, -30),
                }
            )
            events.append(wifi_event)

        # Probabilistically add cell tower event
        if self.should_generate_event(0.2):
            cell_event = self.create_base_event(
                user.user_id, device_id, timestamp, "cell_tower", user=user
            )
            cell_event.update(
                {
                    "cell_id": f"CELL_{np.random.randint(1000, 9999)}",
                    "location_type": location_type,
                    "signal_strength_dbm": np.random.uniform(-90, -50),
                }
            )
            events.append(cell_event)

        self.events.extend(events)
        return events

    def get_user_location_summary(self, user_id: str) -> Dict:
        """Get summary of user's location patterns."""
        if user_id not in self.user_locations:
            return {}

        locations = self.user_locations[user_id]
        movement_state = self.user_movement_state.get(user_id, {})

        return {
            "home": locations["home"],
            "work": locations["work"],
            "leisure_locations": len(locations["leisure"]),
            "current_location": movement_state.get("current_location", "unknown"),
            "in_motion": movement_state.get("in_motion", False),
            "total_distance_today_km": np.random.exponential(15),  # Simulated
        }
