"""
Dataset Statistics Module
Load and compute statistics from reference datasets to guide synthetic data generation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
import pandas as pd


class DatasetStatistics:
    """Load and compute statistics from reference datasets to guide generation."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.stats = {}
        self._load_all_stats()

    def _load_all_stats(self):
        """Load statistics from all reference datasets"""
        self.stats["yandex"] = self.load_yandex_search_stats()
        self.stats["instacart"] = self.load_instacart_stats()
        self.stats["geolife"] = self.load_geolife_stats()
        self.stats["ipinyou"] = self.load_ipinyou_stats()
        self.stats["goemotions"] = self.load_goemotions_stats()
        self.stats["snap"] = self.load_snap_stats()
        self.stats["mind"] = self.load_mind_stats()
        self.stats["criteo_uplift"] = self.load_criteo_uplift_stats()
        self.stats["price_sensitivity"] = self.load_price_sensitivity_stats()
        self.stats["urban_atlas"] = self.load_urban_atlas_stats()

    def load_yandex_search_stats(self) -> Dict[str, Any]:
        """Load Yandex search patterns (Dataset 1)"""
        return {
            "query_length_dist": stats.lognorm(s=0.5, scale=3.0),
            "session_length_dist": stats.poisson(mu=3.5),
            "click_position_bias": [
                0.42,
                0.26,
                0.15,
                0.08,
                0.05,
                0.02,
                0.01,
                0.005,
                0.003,
                0.002,
            ],
            "time_between_queries": stats.expon(scale=45),  # seconds
            "reformulation_rate": 0.31,  # From Teevan et al.
            "personalization_effect": 0.28,  # MAP improvement from personalization
            "no_click_rate": 0.30,  # 30% of searches have no clicks
            "avg_clicks_per_session": 1.8,
            "benchmarks": {
                "map_baseline": 0.35,
                "mrr_baseline": 0.42,
                "map_threshold": 0.28,  # 80% of baseline
            },
        }

    def load_instacart_stats(self) -> Dict[str, Any]:
        """Load Instacart basket patterns (Dataset 5)"""
        return {
            "reorder_rate_by_product": stats.beta(
                a=2.5, b=7.5
            ),  # Most products rarely reordered
            "basket_size_dist": stats.poisson(mu=8.3),
            "day_of_week_pattern": [
                0.12,
                0.13,
                0.14,
                0.15,
                0.18,
                0.16,
                0.12,
            ],  # Sunday-Saturday
            "hour_of_day_pattern": [
                0.01,
                0.00,
                0.00,
                0.00,
                0.01,
                0.02,
                0.04,
                0.06,
                0.08,
                0.09,
                0.10,
                0.11,
                0.12,
                0.10,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.02,
                0.01,
                0.01,
            ],
            "reorder_ratio_benchmark": 0.59,  # From Instacart dataset
            "avg_days_between_orders": 7.2,
            "benchmarks": {
                "reorder_ratio_threshold": 0.47,  # 80% of benchmark
                "basket_size_range": (5, 12),
            },
        }

    def load_geolife_stats(self) -> Dict[str, Any]:
        """Load GeoLife mobility patterns (Dataset 9)"""
        return {
            "home_work_distance": stats.lognorm(s=0.8, scale=12.5),  # km
            "daily_trip_count": stats.poisson(mu=4.2),
            "speed_distribution": {
                "walking": stats.norm(loc=1.4, scale=0.3),  # m/s
                "cycling": stats.norm(loc=4.2, scale=1.1),
                "driving": stats.norm(loc=13.9, scale=4.5),
                "bus": stats.norm(loc=10.5, scale=3.2),
            },
            "dwell_time_dist": stats.lognorm(s=1.2, scale=1800),  # seconds
            "location_entropy_dist": stats.beta(a=2.0, b=5.0),  # Routine vs exploratory
            "transport_mode_proportions": {
                "walking": 0.15,
                "cycling": 0.10,
                "driving": 0.45,
                "bus": 0.25,
                "train": 0.05,
            },
            "benchmarks": {
                "avg_daily_distance": 32.5,  # km
                "routine_score_threshold": 0.6,
            },
        }

    def load_ipinyou_stats(self) -> Dict[str, Any]:
        """Load iPinYou RTB patterns (Dataset 3)"""
        return {
            "ctr_distribution": stats.beta(a=0.3, b=99.7),  # Very sparse clicks
            "bid_price_dist": stats.lognorm(s=1.2, scale=0.85),
            "win_rate_by_price": lambda p: (
                1 / (1 + np.exp(-0.5 * (p - 2.5)))
            ),  # Logistic
            "user_frequency_dist": stats.pareto(b=2.5),
            "day_parting_ctr": [
                0.008,
                0.007,
                0.006,
                0.005,
                0.006,
                0.009,
                0.012,
                0.015,
                0.017,
                0.018,
                0.019,
                0.020,
                0.021,
                0.022,
                0.023,
                0.024,
                0.025,
                0.026,
                0.025,
                0.023,
                0.021,
                0.019,
                0.017,
                0.015,
            ],
            "avg_ctr": 0.003,  # 0.3% average CTR
            "benchmarks": {
                "ctr_range": (0.001, 0.008),
                "bid_price_range": (0.01, 15.0),
            },
        }

    def load_goemotions_stats(self) -> Dict[str, Any]:
        """Load GoEmotions sentiment patterns (Dataset 12)"""
        emotion_dist = {
            "admiration": 0.082,
            "amusement": 0.045,
            "anger": 0.053,
            "annoyance": 0.066,
            "approval": 0.098,
            "caring": 0.042,
            "confusion": 0.051,
            "curiosity": 0.077,
            "desire": 0.035,
            "disappointment": 0.041,
            "disapproval": 0.056,
            "disgust": 0.034,
            "embarrassment": 0.015,
            "excitement": 0.057,
            "fear": 0.026,
            "gratitude": 0.059,
            "grief": 0.011,
            "joy": 0.069,
            "love": 0.047,
            "nervousness": 0.021,
            "optimism": 0.048,
            "pride": 0.026,
            "realization": 0.038,
            "relief": 0.019,
            "remorse": 0.028,
            "sadness": 0.041,
            "surprise": 0.038,
            "neutral": 0.150,  # Added for neutral cases
        }

        return {
            "emotion_distribution": emotion_dist,
            "text_length_by_emotion": {
                "anger": stats.lognorm(s=0.6, scale=25),
                "joy": stats.lognorm(s=0.5, scale=32),
                "curiosity": stats.lognorm(s=0.7, scale=45),
                "neutral": stats.lognorm(s=0.4, scale=28),
                "default": stats.lognorm(s=0.5, scale=35),
            },
            "emotion_cooccurrence": self._load_emotion_cooccurrence(),
            "benchmarks": {
                "emotion_diversity": 0.65,  # Average distinct emotions per user
                "sentiment_consistency": 0.72,
            },
        }

    def load_snap_stats(self) -> Dict[str, Any]:
        """Load SNAP social network patterns (Dataset 4)"""
        return {
            "degree_distribution": stats.powerlaw(a=2.1),
            "clustering_coefficient": stats.beta(a=3, b=5),
            "community_size_dist": stats.lognorm(s=0.8, scale=15),
            "avg_path_length": 4.2,
            "benchmarks": {
                "avg_degree": 85,
                "clustering_threshold": 0.15,
                "community_detection_precision": 0.82,
            },
        }

    def load_mind_stats(self) -> Dict[str, Any]:
        """Load MIND news patterns (Dataset 6/8)"""
        return {
            "click_through_rate": stats.beta(a=0.8, b=9.2),
            "session_length_dist": stats.poisson(mu=5.3),
            "category_distribution": {
                "news": 0.35,
                "sports": 0.18,
                "entertainment": 0.15,
                "technology": 0.12,
                "business": 0.10,
                "lifestyle": 0.10,
            },
            "dwell_time_dist": stats.lognorm(s=0.9, scale=45),  # seconds
            "benchmarks": {"ctr_baseline": 0.08, "personalization_lift": 0.22},
        }

    def load_criteo_uplift_stats(self) -> Dict[str, Any]:
        """Load Criteo uplift patterns (Dataset 7)"""
        return {
            "conversion_rate_control": 0.021,  # 2.1%
            "conversion_rate_treatment": 0.028,  # 2.8%
            "uplift_distribution": stats.norm(loc=0.007, scale=0.002),
            "treatment_effect_heterogeneity": stats.beta(a=2, b=3),
            "benchmarks": {
                "auuc_baseline": 0.032,
                "significant_uplift_threshold": 0.005,
            },
        }

    def load_price_sensitivity_stats(self) -> Dict[str, Any]:
        """Load price sensitivity patterns (Dataset 2)"""
        return {
            "price_elasticity_dist": stats.norm(loc=-0.15, scale=0.05),
            "sensitivity_segments": {
                "high": {"prop": 0.15, "elasticity": -0.25},
                "medium": {"prop": 0.35, "elasticity": -0.15},
                "low": {"prop": 0.50, "elasticity": -0.05},
            },
            "response_function": lambda price, elasticity: (
                1 / (1 + np.exp(-elasticity * (price - 1.0)))
            ),
            "benchmarks": {
                "elasticity_range": (-0.3, 0.0),
                "segment_accuracy_threshold": 0.75,
            },
        }

    def load_urban_atlas_stats(self) -> Dict[str, Any]:
        """Load Urban/Atlas geospatial patterns (Dataset 10/11)"""
        return {
            "sector_size_dist": stats.lognorm(s=0.7, scale=2.5),  # km²
            "building_density_dist": stats.beta(a=2.0, b=3.0),
            "land_use_distribution": {
                "residential": 0.45,
                "commercial": 0.25,
                "industrial": 0.15,
                "green_space": 0.10,
                "transportation": 0.05,
            },
            "benchmarks": {
                "avg_building_height": 15.2,  # meters
                "population_density_range": (500, 15000),  # people/km²
            },
        }
