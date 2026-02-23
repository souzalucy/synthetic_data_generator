"""Device stitching and graph construction."""

import json
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class GraphStitcher:
    """Probabilistic device matching and user stitching."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize graph stitcher.
        
        Args:
            confidence_threshold: Minimum confidence for linking (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.device_clusters = {}  # device_id -> user_id
        self.confidence_scores = {}  # (device_id, device_id) -> confidence
    
    def stitch_events(self, events: List[Dict]) -> Dict:
        """Probabilistically stitch devices to users.
        
        Args:
            events: List of raw events
            
        Returns:
            Dictionary with stitched data and metrics
        """
        # 1. Extract explicit login signals (100% confidence)
        explicit_links = self._extract_explicit_links(events)
        
        # 2. IP-based matching (70% confidence if multiple events same IP+time)
        ip_links = self._ip_based_matching(events)
        
        # 3. GPS distance matching (60% confidence)
        gps_links = self._gps_based_matching(events)
        
        # 4. Behavioral fingerprinting (50% confidence)
        behavior_links = self._behavioral_fingerprinting(events)
        
        # 5. Resolve conflicts and create final clusters
        final_clusters = self._resolve_clusters(explicit_links, ip_links, gps_links, behavior_links)
        
        # 6. Compute statistics
        metrics = self._compute_metrics(final_clusters, events)
        
        return {
            "device_clusters": final_clusters,
            "metrics": metrics,
            "confidence_threshold": self.confidence_threshold,
        }
    
    def _extract_explicit_links(self, events: List[Dict]) -> Dict[str, str]:
        """Extract 100% confident linkages from login events."""
        links = {}
        
        for event in events:
            if event.get("event_type") == "login":
                device_id = event.get("device_id")
                user_id = event.get("user_id")
                if device_id and user_id:
                    links[device_id] = user_id
        
        return links
    
    def _ip_based_matching(self, events: List[Dict]) -> Dict[Tuple[str, str], float]:
        """Match devices by IP similarity within 1 hour."""
        links = {}
        ip_events = defaultdict(list)
        
        # Group by IP
        for event in events:
            ip = event.get("ip_address")
            if ip:
                ip_events[ip].append(event)
        
        # Within each IP group, check device pairs
        for ip, group in ip_events.items():
            devices = set(e.get("device_id") for e in group if e.get("device_id"))
            
            for dev1 in devices:
                for dev2 in devices:
                    if dev1 < dev2:  # Avoid duplicates
                        # Check if events occur within 1 hour for same device
                        time_diff = self._compute_time_diff(
                            [e for e in group if e.get("device_id") == dev1],
                            [e for e in group if e.get("device_id") == dev2]
                        )
                        
                        if time_diff < 3600:  # 1 hour in seconds
                            links[(dev1, dev2)] = 0.7
        
        return links
    
    def _gps_based_matching(self, events: List[Dict]) -> Dict[Tuple[str, str], float]:
        """Match devices by GPS proximity (<100m)."""
        links = {}
        gps_events = [e for e in events if e.get("event_type") in ["gps_update", "wifi_connection"]]
        
        devices_with_gps = set(e.get("device_id") for e in gps_events if e.get("device_id"))
        
        for dev1 in devices_with_gps:
            for dev2 in devices_with_gps:
                if dev1 < dev2:
                    # Get GPS coordinates
                    coords1 = [e for e in gps_events if e.get("device_id") == dev1 and "latitude" in e]
                    coords2 = [e for e in gps_events if e.get("device_id") == dev2 and "latitude" in e]
                    
                    if coords1 and coords2:
                        # Compute average distance
                        distances = []
                        for c1 in coords1:
                            for c2 in coords2:
                                dist = np.sqrt(
                                    (c1["latitude"] - c2["latitude"])**2 +
                                    (c1["longitude"] - c2["longitude"])**2
                                ) * 111000  # Convert degrees to meters (~111 km per degree)
                                distances.append(dist)
                        
                        avg_dist = np.mean(distances)
                        if avg_dist < 100:  # Within 100m
                            links[(dev1, dev2)] = 0.6
        
        return links
    
    def _behavioral_fingerprinting(self, events: List[Dict]) -> Dict[Tuple[str, str], float]:
        """Match devices by behavior similarity."""
        links = {}
        
        # Build behavior profiles
        behaviors = defaultdict(lambda: {"categories": [], "times": []})
        for event in events:
            device = event.get("device_id")
            if device:
                if "category" in event:
                    behaviors[device]["categories"].append(event["category"])
                if "timestamp" in event:
                    hour = int(event["timestamp"].split("T")[1].split(":")[0])
                    behaviors[device]["times"].append(hour)
        
        # Compute similarity between devices
        devices = list(behaviors.keys())
        for i, dev1 in enumerate(devices):
            for dev2 in devices[i+1:]:
                # Jaccard similarity on category sets
                cats1 = set(behaviors[dev1]["categories"])
                cats2 = set(behaviors[dev2]["categories"])
                
                if cats1 and cats2:
                    overlap = len(cats1 & cats2)
                    union = len(cats1 | cats2)
                    jaccard = overlap / union if union > 0 else 0
                    
                    # Time overlap
                    time1 = set(behaviors[dev1]["times"])
                    time2 = set(behaviors[dev2]["times"])
                    time_overlap = len(time1 & time2) / min(len(time1), len(time2)) if min(len(time1), len(time2)) > 0 else 0
                    
                    # Combined similarity
                    similarity = jaccard * 0.5 + time_overlap * 0.5
                    
                    if similarity > 0.5:
                        links[(dev1, dev2)] = 0.5 * similarity
        
        return links
    
    def _resolve_clusters(self, explicit: Dict, ip_links: Dict, gps_links: Dict, 
                         behavior_links: Dict) -> Dict[str, List[str]]:
        """Resolve device clusters using union-find."""
        # Start with explicit links
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Add explicit links first
        for device, user in explicit.items():
            union(device, user)
        
        # Add IP links (only if confidence high enough)
        for (dev1, dev2), conf in ip_links.items():
            if conf >= self.confidence_threshold:
                union(dev1, dev2)
        
        # Add GPS links
        for (dev1, dev2), conf in gps_links.items():
            if conf >= self.confidence_threshold:
                union(dev1, dev2)
        
        # Add behavior links
        for (dev1, dev2), conf in behavior_links.items():
            if conf >= self.confidence_threshold:
                union(dev1, dev2)
        
        # Build final clusters
        clusters = defaultdict(list)
        for device in parent.keys():
            root = find(device)
            clusters[root].append(device)
        
        return dict(clusters)
    
    def _compute_time_diff(self, events1: List[Dict], events2: List[Dict]) -> float:
        """Compute minimum time difference between event groups."""
        if not events1 or not events2:
            return float('inf')
        
        times1 = [self._parse_timestamp(e.get("timestamp", "")) for e in events1]
        times2 = [self._parse_timestamp(e.get("timestamp", "")) for e in events2]
        
        min_diff = float('inf')
        for t1 in times1:
            for t2 in times2:
                if t1 and t2:
                    diff = abs((t2 - t1).total_seconds())
                    min_diff = min(min_diff, diff)
        
        return min_diff
    
    def _parse_timestamp(self, ts: str) -> object:
        """Parse ISO timestamp."""
        from datetime import datetime
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except:
            return None
    
    def _compute_metrics(self, clusters: Dict, events: List[Dict]) -> Dict:
        """Compute stitching quality metrics."""
        n_devices = len(set(e.get("device_id") for e in events if e.get("device_id")))
        n_clusters = len(clusters)
        
        return {
            "n_devices": n_devices,
            "n_clusters": n_clusters,
            "avg_cluster_size": n_devices / n_clusters if n_clusters > 0 else 0,
        }
