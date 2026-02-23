"""Geographic stream for location-based events."""

import numpy as np
from typing import List, Dict
from .base_stream import BaseStream


class GeoStream(BaseStream):
    """Generates location-based events (mobile only)."""
    
    def __init__(self, config: Dict, state_manager, taxonomy: Dict):
        """Initialize geo stream."""
        super().__init__(config, state_manager, taxonomy, "GEO_STREAM")
        self.sectors = list(taxonomy["sectors"].keys())
        # Assign home/work location per user
        self.user_locations = {}
    
    def generate_event(self, user, timestamp: str, probabilities: Dict) -> List[Dict]:
        """Generate geo events for a user.
        
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
        
        if self.should_generate_event(probabilities["geo_update"]):
            device_id = np.random.choice([d.device_id for d in mobile_devices])
            
            # Get or create user location
            if user.user_id not in self.user_locations:
                # Initialize home location
                home_lat = np.random.uniform(40.6, 40.9)  # NYC area
                home_lon = np.random.uniform(-74.2, -73.8)
                self.user_locations[user.user_id] = {
                    "home": (home_lat, home_lon),
                    "work": (home_lat + np.random.uniform(-0.05, 0.05), 
                            home_lon + np.random.uniform(-0.05, 0.05)),
                }
            
            locations = self.user_locations[user.user_id]
            
            # Determine location type based on time
            from datetime import datetime
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = ts.hour
            weekday = ts.weekday()
            
            if weekday < 5:  # Weekday
                if 0 <= hour < 7 or 22 <= hour < 24:
                    location_type = "home"
                    base_lat, base_lon = locations["home"]
                elif 9 <= hour < 17:
                    location_type = "work"
                    base_lat, base_lon = locations["work"]
                elif hour in [8, 18]:
                    location_type = "commute"
                    # Interpolate between home and work
                    base_lat = (locations["home"][0] + locations["work"][0]) / 2
                    base_lon = (locations["home"][1] + locations["work"][1]) / 2
                else:
                    location_type = "other"
                    base_lat = locations["home"][0]
                    base_lon = locations["home"][1]
            else:  # Weekend
                if 0 <= hour < 9 or 22 <= hour < 24:
                    location_type = "home"
                    base_lat, base_lon = locations["home"]
                else:
                    location_type = "other"
                    base_lat = locations["home"][0]
                    base_lon = locations["home"][1]
            
            # Add GPS noise
            lat = base_lat + np.random.normal(0, 0.001)
            lon = base_lon + np.random.normal(0, 0.001)
            
            gps_event = self.create_base_event(user.user_id, device_id, timestamp, "gps_update")
            gps_event.update({
                "latitude": lat,
                "longitude": lon,
                "accuracy_meters": 15.3,
                "location_type": location_type,
                "home_sector": user.home_sector,
            })
            events.append(gps_event)
            
            # Probabilistically add WiFi event
            if self.should_generate_event(0.3):
                wifi_event = self.create_base_event(user.user_id, device_id, timestamp, "wifi_connection")
                ssid = f"WiFi_{location_type}_{np.random.randint(1, 100)}"
                wifi_event.update({
                    "ssid": ssid,
                    "location_type": location_type,
                })
                events.append(wifi_event)
        
        self.events.extend(events)
        return events
