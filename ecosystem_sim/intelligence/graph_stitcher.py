"""Device stitching and graph construction with SNAP network structures."""

import json
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
import networkx as nx
from datetime import datetime, timedelta
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class GraphStitcher:
    """Probabilistic device matching and user stitching with SNAP network patterns."""

    def __init__(
        self, confidence_threshold: float = 0.7, network_priors: Optional[Dict] = None
    ):
        """Initialize graph stitcher with SNAP network patterns."""
        self.confidence_threshold = confidence_threshold
        self.network_priors = network_priors or {}

        # SNAP network patterns
        self.degree_distribution = self.network_priors.get("degree_distribution", {})
        self.clustering_coefficient = self.network_priors.get(
            "clustering_coefficient", 0.1
        )

        # Network graphs
        self.combined_graph = nx.Graph()

        # Inferred user_id mappings (Layer 3)
        self.inferred_user_map: Dict[str, str] = {}  # event_id -> inferred_user_id

    def stitch_events(self, events: List[Dict]) -> Dict:
        """Probabilistically stitch devices to users using SNAP network patterns."""
        # Extract links from different evidence sources (parallelized)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._extract_explicit_links, events): "explicit",
                executor.submit(self._ip_based_matching, events): "ip",
                executor.submit(self._gps_based_matching, events): "gps",
                executor.submit(self._behavioral_fingerprinting, events): "behavior",
            }
            results = {}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  [5a] Matching sources",
                leave=False,
            ):
                name = futures[future]
                results[name] = future.result()

        explicit_links = results["explicit"]
        ip_links = results["ip"]
        gps_links = results["gps"]
        behavior_links = results["behavior"]

        # Build combined graph
        self._build_combined_graph(ip_links, gps_links, behavior_links)

        # Apply community detection
        final_clusters = self._apply_community_detection(explicit_links)

        # Infer user_id for unidentified events (Layer 3)
        self._infer_missing_user_ids(events, final_clusters)

        # Compute metrics
        metrics = self._compute_metrics(final_clusters, events)

        return {
            "device_clusters": final_clusters,
            "metrics": metrics,
            "confidence_threshold": self.confidence_threshold,
            "inferred_user_ids": self.inferred_user_map,
        }

    # ------------------------------------------------------------------ #
    #  Layer 3: User ID inference for unidentified events                #
    # ------------------------------------------------------------------ #

    def _infer_missing_user_ids(
        self, events: List[Dict], clusters: Dict[str, List[str]]
    ):
        """Infer user_id for events where user_id is None.

        Strategy:
        1. Build a device->user mapping from known (explicit) links
        2. For each event with missing user_id, check if its device_id
           belongs to a known cluster
        3. If device is in a cluster with known user_id, assign that user_id
        4. For remaining events, use temporal+spatial proximity to known events
        """
        # Build device -> user mapping from clusters that have a real user_id
        device_to_user: Dict[str, str] = {}
        for cluster_id, devices in clusters.items():
            # Check if cluster_id looks like a real user_id (not "cluster_N")
            if not cluster_id.startswith("cluster_"):
                for device in devices:
                    device_to_user[device] = cluster_id

        # Also build from events with explicit user_id
        for event in events:
            uid = event.get("user_id")
            did = event.get("device_id")
            if uid and did:
                device_to_user[did] = uid

        # Pass 1: Direct device-to-user mapping
        for event in events:
            eid = event.get("event_id")
            uid = event.get("user_id")
            did = event.get("device_id")
            if eid and not uid and did and did in device_to_user:
                self.inferred_user_map[eid] = device_to_user[did]

        # Pass 2: Temporal-spatial inference for remaining unidentified events
        unidentified = [
            e
            for e in events
            if e.get("event_id")
            and not e.get("user_id")
            and e.get("event_id") not in self.inferred_user_map
        ]
        identified = [e for e in events if e.get("user_id") and e.get("event_id")]

        for ue in tqdm(unidentified, desc="    [5a.4] User ID inference", leave=False):
            best_match = self._find_best_temporal_match(ue, identified)
            if best_match:
                self.inferred_user_map[ue["event_id"]] = best_match

    def _find_best_temporal_match(
        self, unidentified_event: Dict, identified_events: List[Dict]
    ) -> Optional[str]:
        """Find the most likely user_id for an unidentified event using
        temporal and spatial proximity to known events.

        Uses a 5-minute sliding window: if an unidentified event occurs
        within 5 minutes and 100m of a known user's event, assign that user_id.
        """
        ue_ts = self._parse_timestamp(unidentified_event.get("timestamp", ""))
        if not ue_ts:
            return None

        ue_lat = unidentified_event.get("latitude")
        ue_lon = unidentified_event.get("longitude")
        ue_ip = unidentified_event.get("ip_address", "")
        ue_device = unidentified_event.get("device_id")

        candidates = defaultdict(float)  # user_id -> confidence score

        for ie in identified_events:
            ie_ts = self._parse_timestamp(ie.get("timestamp", ""))
            if not ie_ts:
                continue

            # Temporal proximity: within 5 minutes
            time_diff = abs((ie_ts - ue_ts).total_seconds())
            if time_diff > 300:  # 5 minutes
                continue

            ie_uid = ie["user_id"]
            score = 0.0

            # Temporal score: closer in time = higher score
            score += max(0, 1.0 - time_diff / 300.0) * 0.4

            # Spatial score: same location = higher score
            ie_lat = ie.get("latitude")
            ie_lon = ie.get("longitude")
            if (
                ue_lat is not None
                and ue_lon is not None
                and ie_lat is not None
                and ie_lon is not None
            ):
                dist = self._haversine_distance(ue_lat, ue_lon, ie_lat, ie_lon)
                if dist < 100:  # within 100m
                    score += max(0, 1.0 - dist / 100.0) * 0.3

            # IP proximity: same subnet = higher score
            ie_ip = ie.get("ip_address", "")
            if ue_ip and ie_ip and ue_ip.count(".") >= 2 and ie_ip.count(".") >= 2:
                ue_subnet = ue_ip.rsplit(".", 2)[0]
                ie_subnet = ie_ip.rsplit(".", 2)[0]
                if ue_subnet == ie_subnet:
                    score += 0.2

            # Same device = strong signal
            ie_device = ie.get("device_id")
            if ue_device and ie_device and ue_device == ie_device:
                score += 0.3

            if score > 0:
                candidates[ie_uid] = max(candidates[ie_uid], score)

        if not candidates:
            return None

        # Return the user_id with highest confidence
        best_user = max(candidates, key=candidates.get)
        best_score = candidates[best_user]
        return best_user if best_score >= 0.3 else None

    # ------------------------------------------------------------------ #
    #  Layer 3: Interaction graph construction                           #
    # ------------------------------------------------------------------ #

    def build_interaction_graphs(self, events: List[Dict]) -> Dict[str, Any]:
        """Build temporal interaction graphs between events for each user.

        Creates a graph per user where:
        - Nodes are events
        - Edges connect events that occur within a 5-minute sliding window
        - Edge weights represent temporal proximity (closer = stronger)

        Returns:
            Dict with:
              - "graphs": Dict[user_id, List[Dict]] - edge list per user
              - "metrics": Dict of graph-level statistics
              - "graphml_path": Path to exported GraphML file (if exported)
        """
        # Group events by user_id (using inferred user_ids where available)
        user_events = defaultdict(list)
        for event in events:
            uid = event.get("user_id")
            if not uid:
                # Try inferred mapping
                eid = event.get("event_id")
                uid = self.inferred_user_map.get(str(eid), "unidentified")
            user_events[uid].append(event)

        graphs = {}
        total_nodes = 0
        total_edges = 0

        for uid, ue in tqdm(
            user_events.items(), desc="  [5b] Building interaction graphs", leave=False
        ):
            # Sort by timestamp
            ue_sorted = sorted(
                ue,
                key=lambda e: (
                    self._parse_timestamp(e.get("timestamp", "")) or datetime.min
                ),
            )

            # Build edges within 5-minute sliding window
            edges = []
            for i, e1 in enumerate(ue_sorted):
                t1 = self._parse_timestamp(e1.get("timestamp", ""))
                if not t1:
                    continue
                for j in range(i + 1, len(ue_sorted)):
                    e2 = ue_sorted[j]
                    t2 = self._parse_timestamp(e2.get("timestamp", ""))
                    if not t2:
                        continue
                    time_diff = abs((t2 - t1).total_seconds())
                    if time_diff > 300:  # 5 minutes
                        # Since sorted, break if gap exceeds window
                        if time_diff > 600:  # 10 min buffer for sparse events
                            break
                        continue

                    # Edge weight: temporal proximity
                    weight = max(0.01, 1.0 - time_diff / 300.0)

                    # Boost weight if same service
                    if e1.get("service") == e2.get("service"):
                        weight = min(1.0, weight * 1.5)

                    edges.append(
                        {
                            "source": e1.get("event_id"),
                            "target": e2.get("event_id"),
                            "weight": round(weight, 4),
                            "time_diff_seconds": time_diff,
                            "source_service": e1.get("service"),
                            "target_service": e2.get("service"),
                            "source_event_type": e1.get("event_type"),
                            "target_event_type": e2.get("event_type"),
                        }
                    )

            graphs[uid] = {
                "n_events": len(ue_sorted),
                "n_edges": len(edges),
                "edges": edges,
                "services": list(
                    set(e.get("service") for e in ue_sorted if e.get("service"))
                ),
            }
            total_nodes += len(ue_sorted)
            total_edges += len(edges)

        # Compute graph-level metrics
        n_users = len(user_events)
        avg_degree = (2 * total_edges) / max(total_nodes, 1)
        edge_density = total_edges / max(total_nodes * (total_nodes - 1) / 2, 1)

        metrics = {
            "n_users_with_graphs": n_users,
            "total_events": total_nodes,
            "total_edges": total_edges,
            "avg_degree": round(avg_degree, 4),
            "edge_density": round(edge_density, 6),
            "avg_events_per_user": round(total_nodes / max(n_users, 1), 2),
            "avg_edges_per_user": round(total_edges / max(n_users, 1), 2),
        }

        return {
            "graphs": graphs,
            "metrics": metrics,
        }

    def export_interaction_graphs(self, graphs_data: Dict, output_path: str):
        """Export interaction graphs as a JSON file with edge lists per user.

        Args:
            graphs_data: Output from build_interaction_graphs()
            output_path: Path to output JSON file
        """
        export = {
            "metrics": graphs_data["metrics"],
            "users": {},
        }

        for uid, graph in graphs_data["graphs"].items():
            export["users"][uid] = {
                "n_events": graph["n_events"],
                "n_edges": graph["n_edges"],
                "services": graph["services"],
                "edges": graph["edges"],
            }

        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)

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
        """Match devices by IP similarity."""
        links = {}
        ip_events = defaultdict(list)

        # Group by IP
        for event in events:
            ip = event.get("ip_address")
            if ip:
                ip_events[ip].append(event)

        for ip, group in tqdm(
            ip_events.items(), desc="    [5a.1] IP matching", leave=False
        ):
            devices = set(e.get("device_id") for e in group if e.get("device_id"))
            device_list = list(devices)

            for i, dev1 in enumerate(device_list):
                for dev2 in device_list[i + 1 :]:
                    time_diff = self._compute_time_diff(
                        [e for e in group if e.get("device_id") == dev1],
                        [e for e in group if e.get("device_id") == dev2],
                    )

                    if time_diff < 3600:  # 1 hour
                        confidence = 0.7
                        if self._ip_subnet_similarity(dev1, dev2, events) > 0.5:
                            confidence += 0.1
                        links[(dev1, dev2)] = min(confidence, 0.95)

        return links

    def _ip_subnet_similarity(self, dev1: str, dev2: str, events: List[Dict]) -> float:
        """Compute similarity based on IP subnet patterns."""
        ips1 = set()
        ips2 = set()

        for event in events:
            if event.get("device_id") == dev1 and event.get("ip_address"):
                ips1.add(event["ip_address"])
            if event.get("device_id") == dev2 and event.get("ip_address"):
                ips2.add(event["ip_address"])

        if not ips1 or not ips2:
            return 0.0

        # Extract subnets (first two octets)
        subnets1 = set(ip.rsplit(".", 2)[0] for ip in ips1 if ip.count(".") >= 2)
        subnets2 = set(ip.rsplit(".", 2)[0] for ip in ips2 if ip.count(".") >= 2)

        overlap = len(subnets1 & subnets2)
        union = len(subnets1 | subnets2)

        return overlap / union if union > 0 else 0.0

    def _gps_based_matching(self, events: List[Dict]) -> Dict[Tuple[str, str], float]:
        """Match devices by GPS proximity."""
        links = {}
        gps_events = [
            e
            for e in events
            if e.get("event_type") in ["gps_update", "wifi_connection"]
        ]

        if not gps_events:
            return links

        device_coords = defaultdict(list)
        for event in gps_events:
            device = event.get("device_id")
            if device and "latitude" in event and "longitude" in event:
                device_coords[device].append((event["latitude"], event["longitude"]))

        devices = list(device_coords.keys())

        for i, dev1 in enumerate(
            tqdm(devices, desc="    [5a.2] GPS matching", leave=False)
        ):
            for dev2 in devices[i + 1 :]:
                coords1 = device_coords[dev1]
                coords2 = device_coords[dev2]

                if coords1 and coords2:
                    # Compute minimum distance
                    min_dist = float("inf")
                    for lat1, lon1 in coords1:
                        for lat2, lon2 in coords2:
                            dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                            min_dist = min(min_dist, dist)

                    # Confidence based on distance
                    if min_dist < 50:
                        confidence = 0.8
                    elif min_dist < 100:
                        confidence = 0.6
                    elif min_dist < 200:
                        confidence = 0.4
                    else:
                        confidence = 0.0

                    if confidence > 0:
                        links[(dev1, dev2)] = confidence

        return links

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance in meters."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _behavioral_fingerprinting(
        self, events: List[Dict]
    ) -> Dict[Tuple[str, str], float]:
        """Match devices by behavior similarity."""
        links = {}
        behaviors = defaultdict(lambda: {"categories": [], "times": []})

        for event in events:
            device = event.get("device_id")
            if device:
                if "category" in event:
                    behaviors[device]["categories"].append(event["category"])
                if "timestamp" in event:
                    hour = int(event["timestamp"].split("T")[1].split(":")[0])
                    behaviors[device]["times"].append(hour)

        devices = list(behaviors.keys())
        for i, dev1 in enumerate(
            tqdm(devices, desc="    [5a.3] Behavior matching", leave=False)
        ):
            for dev2 in devices[i + 1 :]:
                # Jaccard similarity on categories
                cats1 = set(behaviors[dev1]["categories"])
                cats2 = set(behaviors[dev2]["categories"])

                if cats1 and cats2:
                    overlap = len(cats1 & cats2)
                    union = len(cats1 | cats2)
                    similarity = overlap / union if union > 0 else 0

                    if similarity > 0.4:
                        links[(dev1, dev2)] = 0.5 * similarity

        return links

    def _build_combined_graph(
        self, ip_links: Dict, gps_links: Dict, behavior_links: Dict
    ):
        """Build combined graph from all evidence sources."""
        self.combined_graph = nx.Graph()

        # Add IP links
        for (dev1, dev2), conf in ip_links.items():
            if conf >= self.confidence_threshold:
                self.combined_graph.add_edge(dev1, dev2, weight=conf, source="ip")

        # Add GPS links
        for (dev1, dev2), conf in gps_links.items():
            if conf >= self.confidence_threshold:
                self.combined_graph.add_edge(dev1, dev2, weight=conf, source="gps")

        # Add behavior links
        for (dev1, dev2), conf in behavior_links.items():
            if conf >= self.confidence_threshold:
                self.combined_graph.add_edge(dev1, dev2, weight=conf, source="behavior")

    def _apply_community_detection(self, explicit_links: Dict) -> Dict[str, List[str]]:
        """Apply community detection to resolve clusters."""
        # Start with explicit links
        clusters = {}

        # Add explicit links first
        for device, user in explicit_links.items():
            if user not in clusters:
                clusters[user] = []
            clusters[user].append(device)

        # Apply connected components on combined graph
        if self.combined_graph.nodes():
            components = list(nx.connected_components(self.combined_graph))

            # Merge components with existing clusters
            for comp in components:
                if len(comp) > 1:
                    # Check if any device already in a cluster
                    existing_cluster = None
                    for device in comp:
                        for cluster_id, devices in clusters.items():
                            if device in devices:
                                existing_cluster = cluster_id
                                break
                        if existing_cluster:
                            break

                    if existing_cluster:
                        # Add all devices to existing cluster
                        clusters[existing_cluster].extend(list(comp))
                    else:
                        # Create new cluster
                        cluster_id = f"cluster_{len(clusters)}"
                        clusters[cluster_id] = list(comp)

        return clusters

    def _compute_time_diff(self, events1: List[Dict], events2: List[Dict]) -> float:
        """Compute minimum time difference between event groups."""
        if not events1 or not events2:
            return float("inf")

        times1 = [self._parse_timestamp(e.get("timestamp", "")) for e in events1]
        times2 = [self._parse_timestamp(e.get("timestamp", "")) for e in events2]

        min_diff = float("inf")
        for t1 in times1:
            for t2 in times2:
                if t1 and t2:
                    diff = abs((t2 - t1).total_seconds())
                    min_diff = min(min_diff, diff)

        return min_diff

    def _parse_timestamp(self, ts: str):
        """Parse ISO timestamp."""
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            return None

    def _compute_metrics(self, clusters: Dict, events: List[Dict]) -> Dict:
        """Compute stitching quality metrics."""
        n_devices = len(set(e.get("device_id") for e in events if e.get("device_id")))
        n_clusters = len(clusters)

        # Compute cluster sizes
        cluster_sizes = [len(devices) for devices in clusters.values()]

        return {
            "n_devices": n_devices,
            "n_clusters": n_clusters,
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "clustering_coefficient": self.clustering_coefficient,
        }
