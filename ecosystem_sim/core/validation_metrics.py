"""Dataset-specific validation metrics for synthetic data quality."""

import numpy as np
from typing import List, Dict
import json
from pathlib import Path
import math


class DatasetValidator:
    """Validate synthetic data against real dataset benchmarks."""

    def __init__(
        self, priors_path: str = "ecosystem_sim/data/priors/compiled_priors.json"
    ):
        """Initialize validator with dataset priors."""
        self.priors_path = Path(priors_path)
        self.priors = self._load_priors()

        # Dataset-specific validation thresholds
        self.validation_thresholds = {
            "distribution_similarity": 0.8,  # Minimum KL divergence similarity
            "statistical_power": 0.8,  # Minimum statistical power
            "effect_size_match": 0.7,  # Minimum effect size correlation
            "temporal_pattern_match": 0.6,  # Minimum temporal pattern similarity
        }

    def _load_priors(self) -> Dict:
        """Load compiled dataset priors."""
        if self.priors_path.exists():
            with open(self.priors_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def validate_yandex_search_patterns(self, search_events: List[Dict]) -> Dict:
        """Validate Yandex search patterns."""
        if not search_events:
            return {"status": "no_data", "score": 0.0}

        # Extract metrics from search events
        query_lengths = []
        click_positions = []
        session_lengths = []
        reformulation_rates = []

        current_session = []
        for event in search_events:
            if event.get("event_type") == "search_query":
                query = event.get("query", "")
                query_lengths.append(len(query.split()))

                if "click_position" in event:
                    click_positions.append(event["click_position"])

                current_session.append(event)
            elif event.get("event_type") == "session_end":
                if current_session:
                    session_lengths.append(len(current_session))

                    # Check for query reformulations
                    queries = [
                        e.get("query", "")
                        for e in current_session
                        if e.get("event_type") == "search_query"
                    ]
                    if len(queries) > 1:
                        reformulations = sum(
                            1
                            for i in range(1, len(queries))
                            if queries[i] != queries[i - 1]
                        )
                        reformulation_rates.append(reformulations / len(queries))

                current_session = []

        # Get Yandex benchmarks from priors
        yandex_benchmarks = self.priors.get("layer_2_streams", {}).get("search", {})

        # Calculate validation scores
        scores = {}

        # Query length distribution
        if query_lengths and "query_length_distribution" in yandex_benchmarks:
            bench_mean = yandex_benchmarks["query_length_distribution"].get("mean", 2.8)
            bench_std = yandex_benchmarks["query_length_distribution"].get("std", 1.4)

            actual_mean = np.mean(query_lengths)
            actual_std = np.std(query_lengths)

            scores["query_length_match"] = self._calculate_distribution_match(
                actual_mean, actual_std, bench_mean, bench_std
            )

        # Click position bias (position 1 should have highest clicks)
        if click_positions:
            position_counts = {}
            for pos in click_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1

            # Check if position 1 has most clicks
            if position_counts:
                max_pos = max(position_counts.items(), key=lambda x: x[1])[0]
                scores["position_bias_correct"] = 1.0 if max_pos == 1 else 0.5

        # Session length distribution
        if session_lengths:
            bench_session_mean = 3.2  # Yandex average
            actual_mean = np.mean(session_lengths)
            scores["session_length_match"] = 1.0 - min(
                1.0, abs(actual_mean - bench_session_mean) / bench_session_mean
            )

        # Overall score
        if scores:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = 0.0

        return {
            "status": "validated",
            "overall_score": overall_score,
            "component_scores": scores,
            "sample_size": len(search_events),
            "benchmarks_used": list(yandex_benchmarks.keys()),
        }

    def validate_instacart_commerce_patterns(self, commerce_events: List[Dict]) -> Dict:
        """Validate Instacart commerce patterns."""
        if not commerce_events:
            return {"status": "no_data", "score": 0.0}

        # Extract metrics
        purchase_events = [
            e for e in commerce_events if e.get("event_type") == "purchase"
        ]
        basket_sizes = []
        reorder_flags = []
        temporal_patterns = []

        for event in purchase_events:
            if "basket_size" in event:
                basket_sizes.append(event["basket_size"])
            if "is_reorder" in event:
                reorder_flags.append(event["is_reorder"])
            if "timestamp" in event:
                # Extract hour of day
                try:
                    hour = int(event["timestamp"].split("T")[1].split(":")[0])
                    temporal_patterns.append(hour)
                except (IndexError, ValueError):
                    pass

        # Get Instacart benchmarks
        instacart_benchmarks = self.priors.get("layer_2_streams", {}).get(
            "commerce", {}
        )

        scores = {}

        # Basket size distribution
        if basket_sizes and "basket_size_distribution" in instacart_benchmarks:
            bench_mean = instacart_benchmarks["basket_size_distribution"].get(
                "mean", 8.7
            )
            bench_std = instacart_benchmarks["basket_size_distribution"].get("std", 6.3)

            actual_mean = np.mean(basket_sizes)
            actual_std = np.std(basket_sizes)

            scores["basket_size_match"] = self._calculate_distribution_match(
                actual_mean, actual_std, bench_mean, bench_std
            )

        # Reorder rate
        if reorder_flags:
            actual_reorder_rate = sum(reorder_flags) / len(reorder_flags)
            bench_reorder_rate = instacart_benchmarks.get("reorder_probability", 0.62)

            scores["reorder_rate_match"] = 1.0 - min(
                1.0, abs(actual_reorder_rate - bench_reorder_rate) / bench_reorder_rate
            )

        # Temporal patterns (purchase hours)
        if temporal_patterns:
            # Check if purchases follow typical patterns (more during day, less at night)
            day_purchases = sum(1 for h in temporal_patterns if 8 <= h <= 20)
            night_purchases = len(temporal_patterns) - day_purchases

            if len(temporal_patterns) > 0:
                day_ratio = day_purchases / len(temporal_patterns)
                # Expect > 70% during daytime
                scores["temporal_pattern"] = min(1.0, day_ratio / 0.7)

        # Overall score
        if scores:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = 0.0

        return {
            "status": "validated",
            "overall_score": overall_score,
            "component_scores": scores,
            "purchase_count": len(purchase_events),
            "benchmarks_used": list(instacart_benchmarks.keys()),
        }

    def validate_geolife_movement_patterns(self, geo_events: List[Dict]) -> Dict:
        """Validate GeoLife GPS movement patterns."""
        if not geo_events:
            return {"status": "no_data", "score": 0.0}

        gps_events = [e for e in geo_events if e.get("event_type") == "gps_update"]

        if len(gps_events) < 10:
            return {"status": "insufficient_data", "score": 0.0}

        # Extract coordinates and calculate movements
        latitudes = []
        longitudes = []
        speeds = []

        prev_lat = None
        prev_lon = None
        prev_time = None

        for event in gps_events:
            lat = event.get("latitude")
            lon = event.get("longitude")
            timestamp = event.get("timestamp")

            if lat is not None and lon is not None:
                latitudes.append(lat)
                longitudes.append(lon)

                if (
                    prev_lat is not None
                    and prev_lon is not None
                    and timestamp
                    and prev_time
                ):
                    # Calculate distance and speed
                    distance = self._haversine_distance(prev_lat, prev_lon, lat, lon)
                    time_diff = self._parse_time_diff(prev_time, timestamp)

                    if time_diff > 0:
                        speed = distance / time_diff * 3600  # km/h
                        speeds.append(speed)

                prev_lat = lat
                prev_lon = lon
                prev_time = timestamp

        # Get GeoLife benchmarks
        geo_benchmarks = self.priors.get("layer_2_streams", {}).get("geo", {})

        scores = {}

        # Movement speed distribution
        if speeds and "movement_speed_kmh" in geo_benchmarks:
            speed_benchmarks = geo_benchmarks["movement_speed_kmh"]

            # Check walking speed
            if "walking" in speed_benchmarks:
                bench_walking = speed_benchmarks["walking"]
                bench_mean = bench_walking.get("mean", 4.2)
                bench_std = bench_walking.get("std", 1.1)

                # Filter walking speeds (< 10 km/h)
                walking_speeds = [s for s in speeds if s < 10]
                if walking_speeds:
                    actual_mean = np.mean(walking_speeds)
                    actual_std = np.std(walking_speeds)

                    scores["walking_speed_match"] = self._calculate_distribution_match(
                        actual_mean, actual_std, bench_mean, bench_std
                    )

        # Spatial coverage (should have reasonable spread)
        if latitudes and longitudes:
            lat_range = max(latitudes) - min(latitudes)
            lon_range = max(longitudes) - min(longitudes)

            # Reasonable urban movement range (approx 0.1 degrees = ~11 km)
            expected_range = 0.1
            lat_score = 1.0 - min(1.0, abs(lat_range - expected_range) / expected_range)
            lon_score = 1.0 - min(1.0, abs(lon_range - expected_range) / expected_range)

            scores["spatial_coverage"] = (lat_score + lon_score) / 2

        # Overall score
        if scores:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = 0.0

        return {
            "status": "validated",
            "overall_score": overall_score,
            "component_scores": scores,
            "gps_points": len(gps_events),
            "movement_segments": len(speeds),
        }

    def validate_goemotions_sentiment_patterns(self, social_events: List[Dict]) -> Dict:
        """Validate GoEmotions sentiment patterns."""
        if not social_events:
            return {"status": "no_data", "score": 0.0}

        # Extract emotion data
        emotions = []
        sentiment_valences = []
        ekman_categories = []

        for event in social_events:
            if "sentiment_cue" in event:
                emotions.append(event["sentiment_cue"])
            if "sentiment_valence" in event:
                sentiment_valences.append(event["sentiment_valence"])
            if "ekman_category" in event:
                ekman_categories.append(event["ekman_category"])

        # Get GoEmotions benchmarks
        emotion_benchmarks = self.priors.get("layer_2_streams", {}).get("social", {})

        scores = {}

        # Emotion distribution diversity
        if emotions:
            unique_emotions = set(emotions)
            scores["emotion_diversity"] = min(
                1.0, len(unique_emotions) / 20
            )  # Up to 20 unique emotions

        # Sentiment valence balance
        if sentiment_valences:
            positive = sum(1 for v in sentiment_valences if v == "positive")
            negative = sum(1 for v in sentiment_valences if v == "negative")
            neutral = sum(1 for v in sentiment_valences if v == "neutral")
            total = len(sentiment_valences)

            if total > 0:
                # Real social media has more positive than negative
                positive_ratio = positive / total
                # Expect ~40-60% positive
                if 0.4 <= positive_ratio <= 0.6:
                    scores["sentiment_balance"] = 1.0
                else:
                    scores["sentiment_balance"] = 1.0 - min(
                        1.0, abs(positive_ratio - 0.5) / 0.5
                    )

        # Ekman category distribution
        if ekman_categories:
            category_counts = {}
            for cat in ekman_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # Check if joy is most common (typical in social media)
            if category_counts:
                max_cat = max(category_counts.items(), key=lambda x: x[1])[0]
                scores["dominant_emotion"] = 1.0 if max_cat == "joy" else 0.7

        # Overall score
        if scores:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = 0.0

        return {
            "status": "validated",
            "overall_score": overall_score,
            "component_scores": scores,
            "emotion_samples": len(emotions),
            "unique_emotions": len(set(emotions)) if emotions else 0,
        }

    def validate_snap_network_structures(self, graph_data: Dict) -> Dict:
        """Validate SNAP network structure patterns."""
        if not graph_data or "edges" not in graph_data:
            return {"status": "no_data", "score": 0.0}

        edges = graph_data["edges"]
        nodes = graph_data.get("nodes", [])

        # Calculate network metrics
        degree_distribution = self._calculate_degree_distribution(edges, nodes)

        scores = {}

        # Degree distribution (power law check)
        if len(degree_distribution) > 10:
            degrees = list(degree_distribution.values())

            # Check for power law characteristics (many low-degree, few high-degree)
            sorted_degrees = sorted(degrees, reverse=True)
            if len(sorted_degrees) >= 10:
                # Ratio of top 10% to bottom 10%
                top_10 = sorted_degrees[: len(sorted_degrees) // 10]
                bottom_10 = sorted_degrees[-len(sorted_degrees) // 10 :]

                if sum(bottom_10) > 0:
                    ratio = sum(top_10) / sum(bottom_10)
                    # Real networks have high ratio (> 10)
                    scores["degree_inequality"] = min(1.0, ratio / 20)

        # Clustering coefficient
        clustering = self._calculate_clustering_coefficient(edges, nodes)
        if clustering is not None:
            # Real social networks have clustering ~0.1-0.3
            if 0.1 <= clustering <= 0.3:
                scores["clustering_coefficient"] = 1.0
            else:
                scores["clustering_coefficient"] = 1.0 - min(
                    1.0, abs(clustering - 0.2) / 0.2
                )

        # Overall score
        if scores:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = 0.0

        return {
            "status": "validated",
            "overall_score": overall_score,
            "component_scores": scores,
            "nodes": len(nodes),
            "edges": len(edges),
            "avg_degree": np.mean(list(degree_distribution.values()))
            if degree_distribution
            else 0,
        }

    def _calculate_distribution_match(
        self, actual_mean: float, actual_std: float, bench_mean: float, bench_std: float
    ) -> float:
        """Calculate similarity between distributions."""
        # Compare means
        mean_diff = abs(actual_mean - bench_mean)
        mean_similarity = 1.0 - min(1.0, mean_diff / max(0.1, bench_mean))

        # Compare standard deviations
        std_diff = abs(actual_std - bench_std)
        std_similarity = 1.0 - min(1.0, std_diff / max(0.1, bench_std))

        # Combined score
        return (mean_similarity + std_similarity) / 2

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance in kilometers."""
        R = 6371  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2) ** 2
