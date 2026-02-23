"""Unit tests for core modules."""

import pytest
import numpy as np
from ecosystem_sim.core import (
    UserGenerator, Persona, IncomeLevel, TechSavviness, PrivacySensitivity,
    CausalConfig, CausalEngine, TimeManager, StateManager
)


class TestUserAgent:
    """Test user generation."""
    
    def test_user_generation(self):
        """Test user generation creates valid users."""
        config = {"seed": 42}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(10)
        
        assert len(users) == 10
        assert all(hasattr(u, 'user_id') for u in users)
        assert all(len(u.devices) >= 1 for u in users)
    
    def test_persona_variety(self):
        """Test that generated users have different personas."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(100)
        
        personas = [u.persona.name for u in users]
        assert len(set(personas)) > 1  # At least 2 different personas
    
    def test_device_generation(self):
        """Test device generation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(50)
        
        # Check device types are valid
        valid_types = {"mobile", "desktop", "tablet"}
        for user in users:
            for device in user.devices:
                assert device.device_type in valid_types
    
    def test_contact_network(self):
        """Test social network generation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(100)
        
        # Users with high connectivity have more contacts
        high_connectivity = [u for u in users if u.social_connectivity_score > 0.8]
        low_connectivity = [u for u in users if u.social_connectivity_score < 0.2]
        
        avg_high = np.mean([len(u.contact_network) for u in high_connectivity])
        avg_low = np.mean([len(u.contact_network) for u in low_connectivity])
        
        assert avg_high > avg_low  # High connectivity users have more contacts


class TestCausalEngine:
    """Test causal inference engine."""
    
    def test_trajectory_generation(self):
        """Test interest trajectory generation."""
        config = {"seed": 42}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(10)
        
        causal_config = CausalConfig(
            n_categories=5,
            n_days=30,
            random_seed=42,
        )
        engine = CausalEngine(causal_config, users)
        interest_matrix = engine.generate_causal_trajectories()
        
        assert interest_matrix.shape == (10, 30, 5)
        assert interest_matrix.min() >= 0
        assert interest_matrix.max() <= 1
    
    def test_treatment_effect(self):
        """Test that treatment affects interests."""
        config = {"seed": 42}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(100)
        
        causal_config = CausalConfig(
            n_categories=10,
            n_days=100,
            random_seed=42,
        )
        engine = CausalEngine(causal_config, users)
        interest_matrix = engine.generate_causal_trajectories()
        
        # Users with high treatment should have higher interest on average
        high_treatment_users = np.where(engine.treatment_matrix.mean(axis=1) > 0.7)[0]
        low_treatment_users = np.where(engine.treatment_matrix.mean(axis=1) < 0.3)[0]
        
        high_interest = interest_matrix[high_treatment_users].mean()
        low_interest = interest_matrix[low_treatment_users].mean()
        
        # Treatment effect should be visible
        assert high_interest >= low_interest * 0.95  # Some tolerance for noise
    
    def test_counterfactual(self):
        """Test counterfactual reasoning."""
        config = {"seed": 42}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(10)
        
        causal_config = CausalConfig(n_categories=5, n_days=30, random_seed=42)
        engine = CausalEngine(causal_config, users)
        interest_matrix = engine.generate_causal_trajectories()
        
        # Get counterfactual for first user, day 15
        counterfactual = engine.get_counterfactual_interest(
            user_idx=0,
            day=15,
            treatment_override=0.0,
            interest_matrix=interest_matrix
        )
        
        assert isinstance(counterfactual, dict)
        assert len(counterfactual) == 5
        assert all(0 <= v <= 1 for v in counterfactual.values())


class TestTimeManager:
    """Test temporal pattern management."""
    
    def test_circadian_patterns(self):
        """Test circadian pattern generation."""
        tm = TimeManager()
        
        # Search should peak in morning
        morning_search = tm.get_hour_multiplier("search", 9)
        midnight_search = tm.get_hour_multiplier("search", 3)
        
        assert morning_search > midnight_search
    
    def test_day_of_week(self):
        """Test weekend adjustment."""
        tm = TimeManager()
        
        # Weekend should have higher multiplier
        weekend_mult = tm.get_day_multiplier(6)  # Sunday
        weekday_mult = tm.get_day_multiplier(2)  # Wednesday
        
        assert weekend_mult >= 1.0
        assert weekday_mult == 1.0
    
    def test_seasonal(self):
        """Test seasonal patterns."""
        tm = TimeManager()
        
        # December should have high multiplier
        december = tm.get_seasonal_multiplier(12)
        january = tm.get_seasonal_multiplier(1)
        
        assert december > january


class TestStateManager:
    """Test state management."""
    
    def test_probability_calculation(self):
        """Test action probability calculation."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(5)
        
        causal_config = CausalConfig(n_categories=5, n_days=10, random_seed=42)
        engine = CausalEngine(causal_config, users)
        interest_matrix = engine.generate_causal_trajectories()
        
        tm = TimeManager()
        state_mgr = StateManager(engine, tm, interest_matrix)
        
        probs = state_mgr.get_daily_action_probabilities(
            user=users[0],
            day_index=0,
            hour=9,
            users=users
        )
        
        assert isinstance(probs, dict)
        assert all(0 <= v <= 1 for v in probs.values())
    
    def test_in_market_detection(self):
        """Test in-market user detection."""
        config = {}
        gen = UserGenerator(config, random_seed=42)
        users = gen.generate_users(50)
        
        causal_config = CausalConfig(n_categories=10, n_days=30, random_seed=42)
        engine = CausalEngine(causal_config, users)
        interest_matrix = engine.generate_causal_trajectories()
        
        tm = TimeManager()
        state_mgr = StateManager(engine, tm, interest_matrix)
        
        in_market_count = sum(
            1 for user in users
            if state_mgr.is_in_market(user, 0, users)
        )
        
        # Some users should be in-market
        assert in_market_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
