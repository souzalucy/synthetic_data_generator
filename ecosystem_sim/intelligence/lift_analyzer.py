"""Lift analysis for campaign effectiveness."""

import numpy as np
from typing import List, Dict
from scipy.stats import ttest_ind


class LiftAnalyzer:
    """Counterfactual analysis and lift calculation."""
    
    def __init__(self, causal_engine):
        """Initialize lift analyzer.
        
        Args:
            causal_engine: Causal inference engine
        """
        self.causal_engine = causal_engine
    
    def calculate_lift(self, campaign_id: str, users: List, 
                       interest_matrix: np.ndarray, campaign_day: int,
                       campaign_category: int = None) -> Dict:
        """Calculate campaign lift vs counterfactual.
        
        Args:
            campaign_id: Campaign identifier
            users: List of users
            interest_matrix: Interest trajectory matrix
            campaign_day: Day campaign ran
            campaign_category: Target category index (None = all)
            
        Returns:
            Lift analysis report
        """
        results = {
            "campaign_id": campaign_id,
            "exposed_users": [],
            "control_users": [],
            "lift_percent": 0,
            "p_value": 1.0,
            "auuc": 0,
        }
        
        exposed_purchases = []
        counterfactual_purchases = []
        
        for user_idx, user in enumerate(users):
            # Get actual outcome (with ads)
            actual_interest = interest_matrix[user_idx, campaign_day, :]
            
            if campaign_category is not None:
                actual_purchases = actual_interest[campaign_category]
            else:
                actual_purchases = actual_interest.mean()
            
            # Get counterfactual outcome (no ads)
            exposure = self.causal_engine.treatment_matrix[user_idx, campaign_day]
            treatment_effect = self.causal_engine.treatment_effects[user_idx]
            
            # Counterfactual: reduce interest by treatment effect
            counterfactual_interest = actual_interest - (treatment_effect * exposure)
            counterfactual_interest = np.clip(counterfactual_interest, 0, 1)
            
            if campaign_category is not None:
                counterfactual_purchases = counterfactual_interest[campaign_category]
            else:
                counterfactual_purchases_val = counterfactual_interest.mean()
            
            # Incremental lift
            if campaign_category is not None:
                incremental = actual_purchases - counterfactual_purchases
            else:
                incremental = actual_purchases - counterfactual_purchases_val
            
            # Assume 30% conversion from interest (interest = probability of purchase event)
            actual_conv = actual_purchases * 0.3
            counterfactual_conv = (counterfactual_purchases if campaign_category 
                                  else counterfactual_purchases_val) * 0.3
            
            exposed_purchases.append(actual_conv)
            counterfactual_purchases.append(counterfactual_conv)
            
            results["exposed_users"].append({
                "user_id": user.user_id,
                "actual": float(actual_conv),
                "counterfactual": float(counterfactual_conv),
                "incremental": float(incremental * 0.3),
            })
        
        # Calculate aggregate lift
        total_actual = sum(exposed_purchases)
        total_counterfactual = sum(counterfactual_purchases)
        
        if total_counterfactual > 0:
            results["lift_percent"] = (
                (total_actual - total_counterfactual) / total_counterfactual * 100
            )
        else:
            results["lift_percent"] = 0
        
        # Statistical significance
        if len(exposed_purchases) > 1 and len(counterfactual_purchases) > 1:
            try:
                t_stat, p_value = ttest_ind(exposed_purchases, counterfactual_purchases)
                results["p_value"] = p_value
            except:
                results["p_value"] = 1.0
        
        # AUUC (Area Under Uplift Curve) - simplified
        results["auuc"] = results["lift_percent"] / 100  # Normalized
        
        return results
    
    def run_ab_test_simulation(self, control_users: List, treatment_users: List,
                               interest_matrix: np.ndarray, campaign_day: int) -> Dict:
        """Simulate A/B test outcome.
        
        Args:
            control_users: Control group users
            treatment_users: Treatment group users
            interest_matrix: Interest matrix
            campaign_day: Campaign day
            
        Returns:
            A/B test results
        """
        control_conversions = []
        treatment_conversions = []
        
        for user in control_users:
            user_idx = next(i for i, u in enumerate(control_users) if u.user_id == user.user_id)
            avg_interest = interest_matrix[user_idx, campaign_day, :].mean()
            conversion = avg_interest * 0.2  # Lower baseline
            control_conversions.append(conversion)
        
        for user in treatment_users:
            user_idx = next(i for i, u in enumerate(treatment_users) if u.user_id == user.user_id)
            avg_interest = interest_matrix[user_idx, campaign_day, :].mean()
            conversion = avg_interest * 0.3  # Higher treatment
            treatment_conversions.append(conversion)
        
        control_rate = np.mean(control_conversions) if control_conversions else 0
        treatment_rate = np.mean(treatment_conversions) if treatment_conversions else 0
        
        # Statistical test
        if len(control_conversions) > 1 and len(treatment_conversions) > 1:
            t_stat, p_value = ttest_ind(treatment_conversions, control_conversions)
        else:
            t_stat, p_value = 0, 1.0
        
        return {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "lift_bps": (treatment_rate - control_rate) * 10000,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
