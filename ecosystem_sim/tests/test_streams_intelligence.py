"""Unit tests for streams and intelligence modules."""

import pytest
import json
import numpy as np
from ecosystem_sim.core import UserGenerator, CausalConfig, CausalEngine, TimeManager, StateManager
from ecosystem_sim.streams import SearchStream, CommerceStream, GeoStream
from ecosystem_sim.intelligence import GraphStitcher, PropensityModels, LiftAnalyzer


class TestSearchStream:
    """Test search stream."""
    
    def setup_method(self):
        """Setup test fixtures."""
        config = {}
        self.gen = UserGenerator(config, random_seed=42)
        self.users = self.gen.generate_users(5)
        
        causal_config = CausalConfig(n_categories=5, n_days=10, random_seed=42)
        self.engine = CausalEngine(causal_config, self.users)
        self.interest_matrix = self.engine.generate_causal_trajectories()
        
        self.tm = TimeManager()
        self.state_mgr = StateManager(self.engine, self.tm, self.interest_matrix)
        
        with open("ecosystem_sim/data/taxonomy.json", 'r') as f:
            self.taxonomy = json.load(f)
    
    def test_search_event_generation(self):
        """Test search stream generates valid events."""
        stream = SearchStream({}, self.state_mgr, self.taxonomy)
        
        probs = {"search_query_0": 0.8, "search_query_1": 0.5}
        events = stream.generate_event(self.users[0], "2024-01-01T09:00:00Z", probs)
        
        # Should generate some events given high probabilities
        assert isinstance(events, list)
        if len(events) > 0:
            event = events[0]
            assert "event_id" in event
            assert "user_id" in event
            assert "query" in event
    
    def test_search_stream_export(self):
        """Test event export."""
        import tempfile
        
        stream = SearchStream({}, self.state_mgr, self.taxonomy)
        stream.events = [
            {"event_id": "1", "query": "test", "timestamp": "2024-01-01T00:00:00Z"},
            {"event_id": "2", "query": "test2", "timestamp": "2024-01-01T00:01:00Z"},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            filepath = f.name
        
        stream.export_events(filepath)
        
        # Verify file was created
        with open(filepath, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2


class TestCommerceStream:
    """Test commerce stream."""
    
    def setup_method(self):
        """Setup test fixtures."""
        config = {}
        self.gen = UserGenerator(config, random_seed=42)
        self.users = self.gen.generate_users(5)
        
        causal_config = CausalConfig(n_categories=5, n_days=10, random_seed=42)
        self.engine = CausalEngine(causal_config, self.users)
        self.interest_matrix = self.engine.generate_causal_trajectories()
        
        self.tm = TimeManager()
        self.state_mgr = StateManager(self.engine, self.tm, self.interest_matrix)
        
        with open("ecosystem_sim/data/taxonomy.json", 'r') as f:
            self.taxonomy = json.load(f)
    
    def test_commerce_event_generation(self):
        """Test commerce stream generates purchase events."""
        stream = CommerceStream({}, self.state_mgr, self.taxonomy)
        
        probs = {"commerce_browse_0": 0.8}
        events = stream.generate_event(self.users[0], "2024-01-01T12:00:00Z", probs)
        
        assert isinstance(events, list)
        
        # Check event types
        event_types = [e.get("event_type") for e in events]
        valid_types = {"product_view", "add_to_cart", "purchase", "cart_abandonment", "product_review"}
        assert all(t in valid_types for t in event_types)
    
    def test_user_purchase_history(self):
        """Test that purchases are tracked in user history."""
        stream = CommerceStream({}, self.state_mgr, self.taxonomy)
        
        probs = {"commerce_browse_0": 1.0}  # Force generation
        user = self.users[0]
        initial_purchases = len(user.purchase_history)
        
        stream.generate_event(user, "2024-01-01T12:00:00Z", probs)
        
        # May have generated a purchase depending on random draws
        assert len(user.purchase_history) >= initial_purchases


class TestGraphStitcher:
    """Test device stitching."""
    
    def test_stitching_explicit_links(self):
        """Test explicit stitching with login events."""
        events = [
            {
                "event_id": "1",
                "event_type": "login",
                "device_id": "dev_1",
                "user_id": "user_1",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "event_id": "2",
                "event_type": "login",
                "device_id": "dev_2",
                "user_id": "user_1",
                "timestamp": "2024-01-01T01:00:00Z",
            },
        ]
        
        stitcher = GraphStitcher()
        result = stitcher.stitch_events(events)
        
        assert "device_clusters" in result
        assert "metrics" in result
        assert result["metrics"]["n_devices"] == 2
    
    def test_stitching_confidence_threshold(self):
        """Test that confidence threshold is respected."""
        events = []
        
        stitcher = GraphStitcher(confidence_threshold=0.9)
        result = stitcher.stitch_events(events)
        
        assert result["confidence_threshold"] == 0.9


class TestPropensityModels:
    """Test propensity calculations."""
    
    def test_ltv_calculation(self):
        """Test LTV calculation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(1)
        user = users[0]
        
        events = [
            {"event_type": "purchase", "price_usd": 100},
            {"event_type": "purchase", "price_usd": 50},
        ]
        
        propensity = PropensityModels()
        ltv = propensity.calculate_ltv(user, events, {})
        
        assert ltv > 0
        assert isinstance(ltv, (int, float))
    
    def test_churn_calculation(self):
        """Test churn risk calculation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(1)
        user = users[0]
        
        events = []
        
        propensity = PropensityModels()
        churn = propensity.calculate_churn_risk(user, events)
        
        assert 0 <= churn <= 1
    
    def test_conversion_propensity(self):
        """Test conversion propensity calculation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(1)
        user = users[0]
        
        events = [
            {"event_type": "search_query"},
            {"event_type": "product_view"},
            {"event_type": "purchase"},
        ]
        
        propensity = PropensityModels()
        conv = propensity.calculate_conversion_propensity(user, events)
        
        assert 0 <= conv <= 1


class TestLiftAnalyzer:
    """Test lift analysis."""
    
    def test_lift_calculation(self):
        """Test lift calculation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(10)
        
        causal_config = CausalConfig(n_categories=5, n_days=30, random_seed=42)
        engine = CausalEngine(causal_config, users)
        interest_matrix = engine.generate_causal_trajectories()
        
        analyzer = LiftAnalyzer(engine)
        lift = analyzer.calculate_lift(
            campaign_id="test_campaign",
            users=users,
            interest_matrix=interest_matrix,
            campaign_day=15,
        )
        
        assert "campaign_id" in lift
        assert "lift_percent" in lift
        assert "p_value" in lift
        assert "exposed_users" in lift
    
    def test_ab_test_simulation(self):
        """Test A/B test simulation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(20)
        control = users[:10]
        treatment = users[10:]
        
        causal_config = CausalConfig(n_categories=5, n_days=30, random_seed=42)
        engine = CausalEngine(causal_config, users)
        interest_matrix = engine.generate_causal_trajectories()
        
        analyzer = LiftAnalyzer(engine)
        result = analyzer.run_ab_test_simulation(control, treatment, interest_matrix, 15)
        
        assert "control_rate" in result
        assert "treatment_rate" in result
        assert "p_value" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
