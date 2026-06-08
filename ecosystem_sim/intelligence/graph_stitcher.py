"""Interaction graph construction from event data.

Provides ground-truth device stitching utilities and temporal interaction
graph construction for Layer 3 analysis.

The heuristic/probabilistic device stitching (IP, GPS, behavioral fingerprinting)
has been removed. Ground-truth graph construction is now handled by the
dedicated :class:`GroundTruthGraphBuilder` class in ``ground_truth_graph.py``.
"""

import json
import os
import tempfile
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime, timedelta
import math
from tqdm import tqdm


class GraphStitcher:
    """Interaction graph construction and ground-truth stitching utilities.

    This class now focuses on:
    1. Ground-truth device stitching (deterministic mapping from known events).
    2. Temporal interaction graph construction (Layer 3 event graphs).

    The heuristic/probabilistic device stitching methods (``stitch_events``,
    ``_ip_based_matching``, ``_gps_based_matching``, ``_behavioral_fingerprinting``,
    ``_build_combined_graph``, ``_apply_community_detection``) have been removed.
    Use :class:`GroundTruthGraphBuilder` for deterministic ground-truth graphs.
    """

    def __init__(
        self, confidence_threshold: float = 0.7, network_priors: Optional[Dict] = None
    ):
        """Initialize graph stitcher.

        Args:
            confidence_threshold: Threshold for device stitching confidence
                (kept for backward compatibility, no longer used for heuristics).
            network_priors: Optional SNAP network priors (kept for compatibility).
        """
        self.confidence_threshold = confidence_threshold
        self.network_priors = network_priors or {}

        # SNAP network patterns (kept for backward compatibility)
        self.degree_distribution = self.network_priors.get("degree_distribution", {})
        self.clustering_coefficient = self.network_priors.get(
            "clustering_coefficient", 0.1
        )

        # Inferred user_id mappings (Layer 3)
        self.inferred_user_map: Dict[str, str] = {}  # event_id -> inferred_user_id

    # ------------------------------------------------------------------
    # Ground-truth device stitching (deterministic)
    # ------------------------------------------------------------------

    @staticmethod
    def build_ground_truth_mapping(
        ground_truth_events: List[Dict],
    ) -> Dict[str, str]:
        """Build a deterministic device→user mapping from ground-truth events.

        Unlike the probabilistic stitching in stitch_events(), this method
        simply reads the real user_id from each ground-truth event, producing
        a 100% accurate device-to-user mapping.

        Args:
            ground_truth_events: List of events where ``user_id`` is always
                the real user (never anonymized).

        Returns:
            Dict mapping ``device_id → user_id``. If a device appears in
            multiple events, the last assignment wins (all should agree).
        """
        mapping: Dict[str, str] = {}
        for event in ground_truth_events:
            uid = event.get("user_id")
            did = event.get("device_id")
            if uid and did:
                mapping[did] = uid
        return mapping

    @staticmethod
    def build_ground_truth_clusters(
        ground_truth_events: List[Dict],
    ) -> Dict[str, List[str]]:
        """Build deterministic device clusters from ground-truth events.

        Groups devices by their real user_id, producing the true
        device-to-user cluster assignment.

        Args:
            ground_truth_events: List of events with real user_id.

        Returns:
            Dict mapping ``user_id → [device_id, ...]``.
        """
        clusters: Dict[str, set] = {}
        for event in ground_truth_events:
            uid = event.get("user_id")
            did = event.get("device_id")
            if uid and did:
                clusters.setdefault(uid, set()).add(did)
        return {uid: sorted(devices) for uid, devices in clusters.items()}

    @staticmethod
    def compare_stitching_accuracy(
        stitched_data: Dict,
        ground_truth_events: List[Dict],
    ) -> Dict:
        """Compare stitching results against deterministic ground truth.

        Evaluates how well a stitching algorithm performed by comparing its
        inferred device clusters against the true device-to-user assignment
        from ground-truth events.

        Args:
            stitched_data: Dict containing ``device_clusters`` (or ``clusters``)
                and ``inferred_user_ids``.
            ground_truth_events: List of events with real user_id
                (from ``BaseStream.events_ground_truth``).

        Returns:
            Dict with accuracy metrics:
            - ``accuracy``: fraction of devices assigned to the correct cluster
            - ``precision``: fraction of stitched clusters that are pure
            - ``recall``: fraction of true user clusters recovered
            - ``f1_score``: harmonic mean of precision and recall
            - ``correct_assignments``, ``total_devices``, etc.
        """
        # Build ground-truth device→user mapping
        gt_device_to_user = GraphStitcher.build_ground_truth_mapping(
            ground_truth_events
        )
        gt_clusters = GraphStitcher.build_ground_truth_clusters(ground_truth_events)

        # Build stitched device→cluster mapping
        # Support both "device_clusters" (runtime) and "clusters" (serialized) keys
        stitched_clusters = stitched_data.get(
            "device_clusters",
            stitched_data.get("clusters", {}),
        )
        stitched_device_to_cluster: Dict[str, str] = {}
        for cluster_id, devices in stitched_clusters.items():
            for device in devices:
                stitched_device_to_cluster[device] = cluster_id

        # --- Per-device accuracy ---
        correct = 0
        total = 0
        device_results = []

        for device, true_user in gt_device_to_user.items():
            if device not in stitched_device_to_cluster:
                device_results.append(
                    {
                        "device": device,
                        "true_user": true_user,
                        "correct": False,
                        "reason": "not_stitched",
                    }
                )
                total += 1
                continue

            predicted_cluster = stitched_device_to_cluster[device]
            cluster_devices = stitched_clusters.get(predicted_cluster, [])

            # Check if any device in this cluster maps to the true user
            cluster_users = set(
                gt_device_to_user.get(d)
                for d in cluster_devices
                if d in gt_device_to_user
            )
            is_correct = true_user in cluster_users
            if is_correct:
                correct += 1
            total += 1

            device_results.append(
                {
                    "device": device,
                    "true_user": true_user,
                    "predicted_cluster": predicted_cluster,
                    "correct": is_correct,
                }
            )

        accuracy = correct / total if total > 0 else 0.0

        # --- Cluster-level precision & recall ---
        # Precision: for each stitched cluster, what fraction of its devices
        # belong to the majority user?
        precision_sum = 0.0
        n_stitched_with_devices = 0
        for cluster_id, devices in stitched_clusters.items():
            if not devices:
                continue
            # Count devices per true user in this cluster
            user_counts: Dict[str, int] = {}
            for d in devices:
                u = gt_device_to_user.get(d)
                if u:
                    user_counts[u] = user_counts.get(u, 0) + 1
            if not user_counts:
                continue
            majority_count = max(user_counts.values())
            precision_sum += majority_count / len(devices)
            n_stitched_with_devices += 1

        precision = (
            precision_sum / n_stitched_with_devices
            if n_stitched_with_devices > 0
            else 0.0
        )

        # Recall: for each true user cluster, what fraction of its devices
        # were stitched together (i.e., assigned to the same stitched cluster)?
        recall_sum = 0.0
        n_true_clusters = 0
        for true_user, true_devices in gt_clusters.items():
            if len(true_devices) < 2:
                continue  # single-device users don't test stitching
            # Find which stitched cluster(s) these devices landed in
            assigned_clusters = set(
                stitched_device_to_cluster.get(d)
                for d in true_devices
                if d in stitched_device_to_cluster
            )
            assigned_clusters.discard(None)
            if not assigned_clusters:
                continue
            # If all devices are in one stitched cluster → perfect recall for this user
            # If split across clusters → partial recall
            # We use: fraction of device pairs that ended up together
            n_pairs = len(true_devices) * (len(true_devices) - 1) / 2
            if n_pairs == 0:
                continue
            # Count pairs that stayed together (same stitched cluster)
            together = 0
            for i in range(len(true_devices)):
                for j in range(i + 1, len(true_devices)):
                    c1 = stitched_device_to_cluster.get(true_devices[i])
                    c2 = stitched_device_to_cluster.get(true_devices[j])
                    if c1 and c2 and c1 == c2:
                        together += 1
            recall_sum += together / n_pairs
            n_true_clusters += 1

        recall = recall_sum / n_true_clusters if n_true_clusters > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "correct_assignments": correct,
            "total_devices": total,
            "n_ground_truth_clusters": len(gt_clusters),
            "n_stitched_clusters": len(stitched_clusters),
            "n_inferred_user_ids": len(stitched_data.get("inferred_user_ids", {})),
            "device_results": device_results,
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

    def build_interaction_graphs(
        self, events: List[Dict], edge_buffer_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build temporal interaction graphs between events for each user.

        Creates a graph per user where:
        - Nodes are events
        - Edges connect events that occur within a 5-minute sliding window
        - Edge weights represent temporal proximity (closer = stronger)

        Uses an O(n × w) sliding-window algorithm (deque-based) instead of
        O(n²) nested loops to avoid memory blowup with large event volumes.

        To avoid holding all edges in RAM, edges are streamed to a temporary
        JSONL file on disk (one JSON object per edge). The returned dict
        contains only metrics and a reference to the edge buffer file path.

        Args:
            events: List of anonymized event dicts.
            edge_buffer_path: Optional path for the edge buffer file. If None,
                a temporary file is created automatically.

        Returns:
            Dict with:
              - "edge_buffer_path": Path to the edge buffer JSONL file
              - "metrics": Dict of graph-level statistics
              - "user_meta": Dict[user_id, {n_events, n_edges, services}]
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

        # Open edge buffer file for streaming writes
        if edge_buffer_path is None:
            fd, edge_buffer_path = tempfile.mkstemp(
                suffix=".jsonl", prefix="interaction_edges_"
            )
            os.close(fd)

        user_meta: Dict[str, Dict] = {}
        total_nodes = 0
        total_edges = 0

        with open(edge_buffer_path, "w") as edge_buf:
            for uid, ue in tqdm(
                user_events.items(),
                desc="  [5b] Building interaction graphs",
                leave=False,
            ):
                # Sort by timestamp
                ue_sorted = sorted(
                    ue,
                    key=lambda e: (
                        self._parse_timestamp(e.get("timestamp", "")) or datetime.min
                    ),
                )

                # Build edges using a sliding window (deque) — O(n × w) instead of O(n²)
                # The window holds (timestamp, event) pairs within a 5-minute horizon.
                n_edges_for_user = 0
                window: deque = deque()

                for event in ue_sorted:
                    t = self._parse_timestamp(event.get("timestamp", ""))
                    if not t:
                        continue

                    # Evict events older than 5 minutes from the window
                    while window and (t - window[0][0]).total_seconds() > 300:
                        window.popleft()

                    # Compare the new event against all events currently in the window
                    for w_ts, w_event in window:
                        time_diff = abs((t - w_ts).total_seconds())
                        # time_diff is guaranteed ≤ 300 by the eviction logic above,
                        # but clamp for numerical safety
                        if time_diff > 300:
                            continue

                        # Edge weight: temporal proximity
                        weight = max(0.01, 1.0 - time_diff / 300.0)

                        # Boost weight if same service
                        if event.get("service") == w_event.get("service"):
                            weight = min(1.0, weight * 1.5)

                        # Write edge directly to buffer file — one JSON object per line
                        edge_buf.write(
                            json.dumps(
                                {
                                    "user_id": uid,
                                    "source": w_event.get("event_id"),
                                    "target": event.get("event_id"),
                                    "weight": round(weight, 4),
                                    "time_diff_seconds": time_diff,
                                    "source_service": w_event.get("service"),
                                    "target_service": event.get("service"),
                                    "source_event_type": w_event.get("event_type"),
                                    "target_event_type": event.get("event_type"),
                                }
                            )
                            + "\n"
                        )
                        n_edges_for_user += 1

                    # Add the current event to the window
                    window.append((t, event))

                user_meta[uid] = {
                    "n_events": len(ue_sorted),
                    "n_edges": n_edges_for_user,
                    "services": list(
                        set(e.get("service") for e in ue_sorted if e.get("service"))
                    ),
                }
                total_nodes += len(ue_sorted)
                total_edges += n_edges_for_user

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
            "edge_buffer_path": edge_buffer_path,
            "metrics": metrics,
            "user_meta": user_meta,
        }

    def export_interaction_graphs(self, graphs_data: Dict, output_path: str):
        """Export interaction graphs as a JSON file with edge lists per user.

        Reads edges from the streaming buffer file (JSONL) and assembles the
        final JSON output. The buffer file is deleted after successful export.

        Args:
            graphs_data: Output from build_interaction_graphs() containing
                ``edge_buffer_path``, ``metrics``, and ``user_meta``.
            output_path: Path to output JSON file.
        """
        edge_buffer_path = graphs_data.get("edge_buffer_path")
        if not edge_buffer_path or not os.path.exists(edge_buffer_path):
            raise FileNotFoundError(
                f"Edge buffer file not found: {edge_buffer_path}. "
                "Ensure build_interaction_graphs() was called first."
            )

        # Group edges by user_id while reading from the buffer
        user_edges: Dict[str, List[Dict]] = defaultdict(list)
        with open(edge_buffer_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                edge = json.loads(line)
                uid = edge.pop("user_id")
                user_edges[uid].append(edge)

        # Assemble the export structure
        user_meta = graphs_data.get("user_meta", {})
        export = {
            "metrics": graphs_data["metrics"],
            "users": {},
        }

        for uid, edges in user_edges.items():
            meta = user_meta.get(uid, {})
            export["users"][uid] = {
                "n_events": meta.get("n_events", 0),
                "n_edges": meta.get("n_edges", len(edges)),
                "services": meta.get("services", []),
                "edges": edges,
            }

        # Write final output
        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)

        # Clean up the temporary buffer file
        try:
            os.remove(edge_buffer_path)
        except OSError:
            pass

    def _parse_timestamp(self, ts: str):
        """Parse ISO timestamp."""
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            return None
