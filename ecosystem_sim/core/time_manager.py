"""Time-based pattern management for realistic temporal behavior."""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np


class TimeManager:
    """Manages temporal patterns for event generation."""
    
    def __init__(self, start_date: str = "2024-01-01T00:00:00Z"):
        """Initialize time manager.
        
        Args:
            start_date: ISO format start date
        """
        self.start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        self.current_date = self.start_date
        
        # Define service-specific circadian patterns
        self._init_circadian_patterns()
        self._init_day_of_week_patterns()
        self._init_seasonal_patterns()
    
    def _init_circadian_patterns(self):
        """Define 24-hour activity patterns for each service."""
        # Hour -> [0, 1] multiplier for each service
        self.circadian = {
            "search": {
                0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.2,
                6: 0.4, 7: 0.6, 8: 0.9, 9: 1.0,  # Morning peak
                10: 0.9, 11: 0.8, 12: 0.9, 13: 1.0,  # Midday peak
                14: 0.8, 15: 0.7, 16: 0.7, 17: 0.8,
                18: 0.9, 19: 1.0, 20: 0.9,  # Evening peak
                21: 0.7, 22: 0.4, 23: 0.2,
            },
            "commerce": {
                0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1,
                6: 0.2, 7: 0.3, 8: 0.5, 9: 0.6,
                10: 0.7, 11: 0.8, 12: 1.0,  # Lunch peak
                13: 0.9, 14: 0.8, 15: 0.7, 16: 0.6,
                17: 0.7, 18: 0.8, 19: 1.0, 20: 0.9,  # Evening peak
                21: 0.6, 22: 0.3, 23: 0.1,
            },
            "media": {
                0: 0.2, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.2,
                6: 0.4, 7: 0.7, 8: 0.9,  # Morning commute
                9: 0.7, 10: 0.6, 11: 0.7, 12: 0.8,
                13: 0.7, 14: 0.6, 15: 0.5, 16: 0.6,
                17: 0.8, 18: 1.0, 19: 0.9,  # Evening peak
                20: 0.8, 21: 0.7, 22: 0.5, 23: 0.3,
            },
            "geo": {
                0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.3,
                6: 0.6, 7: 0.8, 8: 0.9,  # Morning commute
                9: 0.6, 10: 0.5, 11: 0.5, 12: 0.6,
                13: 0.7, 14: 0.6, 15: 0.5, 16: 0.6,
                17: 0.9, 18: 0.8,  # Evening commute
                19: 0.5, 20: 0.3, 21: 0.2, 22: 0.1, 23: 0.1,
            },
            "social": {
                0: 0.3, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
                6: 0.3, 7: 0.5, 8: 0.7,
                9: 0.6, 10: 0.7, 11: 0.8, 12: 0.8,
                13: 0.7, 14: 0.6, 15: 0.7, 16: 0.8,
                17: 0.9, 18: 1.0, 19: 0.9,  # Evening peak
                20: 0.8, 21: 0.7, 22: 0.5, 23: 0.4,
            },
            "email": {
                0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1,
                6: 0.3, 7: 0.6, 8: 0.9, 9: 1.0,  # Morning peak
                10: 0.9, 11: 0.8, 12: 0.7, 13: 0.8,
                14: 0.7, 15: 0.8, 16: 0.9, 17: 0.9,
                18: 0.7, 19: 0.5, 20: 0.3, 21: 0.2, 22: 0.1, 23: 0.1,
            },
        }
    
    def _init_day_of_week_patterns(self):
        """Define weekday vs weekend patterns."""
        # Day of week: 0=Monday, 6=Sunday
        # Multiplier adjustment for being weekend
        self.day_multiplier = {
            0: 1.0,   # Monday
            1: 1.0,   # Tuesday
            2: 1.0,   # Wednesday
            3: 1.0,   # Thursday
            4: 1.0,   # Friday
            5: 1.1,   # Saturday (30% higher activity)
            6: 1.15,  # Sunday (more free time)
        }
    
    def _init_seasonal_patterns(self):
        """Define seasonal and holiday patterns."""
        # Month -> multiplier for shopping/travel intent
        self.seasonal = {
            1: 1.1,   # January (New Year resolutions)
            2: 1.0,   # February
            3: 1.1,   # March (Spring)
            4: 1.2,   # April (Easter, spring break)
            5: 1.15,  # May (Mother's Day)
            6: 1.1,   # June (Summer)
            7: 1.3,   # July (Summer travel)
            8: 1.2,   # August (Back to school)
            9: 1.0,   # September
            10: 1.2,  # October (Halloween)
            11: 1.4,  # November (Black Friday, Thanksgiving)
            12: 1.5,  # December (Christmas, New Year)
        }
        
        # Major holidays (month, day)
        self.holidays = [
            (1, 1),    # New Year
            (7, 4),    # July 4th
            (11, 28),  # Thanksgiving (~4th Thursday)
            (12, 25),  # Christmas
        ]
    
    def get_hour_multiplier(self, service: str, hour: int) -> float:
        """Get temporal multiplier for a service at given hour.
        
        Args:
            service: Service name (search, commerce, etc.)
            hour: Hour of day (0-23)
            
        Returns:
            Multiplier [0, 1]
        """
        if service not in self.circadian:
            return 0.5  # Default for unknown services
        
        return self.circadian[service].get(hour, 0.5)
    
    def get_day_multiplier(self, day_of_week: int) -> float:
        """Get multiplier for day of week.
        
        Args:
            day_of_week: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            Multiplier
        """
        return self.day_multiplier.get(day_of_week, 1.0)
    
    def get_seasonal_multiplier(self, month: int) -> float:
        """Get multiplier for month/season.
        
        Args:
            month: Month (1-12)
            
        Returns:
            Multiplier
        """
        return self.seasonal.get(month, 1.0)
    
    def get_combined_multiplier(self, service: str, hour: int, 
                                day_of_week: int, month: int) -> float:
        """Get combined temporal multiplier.
        
        Args:
            service: Service name
            hour: Hour of day
            day_of_week: Day of week (0-6)
            month: Month (1-12)
            
        Returns:
            Combined multiplier
        """
        hour_mult = self.get_hour_multiplier(service, hour)
        day_mult = self.get_day_multiplier(day_of_week)
        seasonal_mult = self.get_seasonal_multiplier(month)
        
        # Combine with relative weights
        combined = hour_mult * 0.6 + day_mult * 0.2 + seasonal_mult * 0.2
        return np.clip(combined, 0.01, 2.0)
    
    def advance_day(self):
        """Move to next day."""
        self.current_date += timedelta(days=1)
    
    def get_day_of_week(self) -> int:
        """Get current day of week (0=Monday, 6=Sunday)."""
        return self.current_date.weekday()
    
    def get_month(self) -> int:
        """Get current month (1-12)."""
        return self.current_date.month
    
    def get_timestamp(self, hour: int = 0, minute: int = 0, 
                     second: int = 0) -> str:
        """Get ISO format timestamp.
        
        Args:
            hour: Hour of day (0-23)
            minute: Minute of hour
            second: Second of minute
            
        Returns:
            ISO 8601 formatted timestamp
        """
        ts = self.current_date.replace(hour=hour, minute=minute, second=second)
        return ts.isoformat() + 'Z'
