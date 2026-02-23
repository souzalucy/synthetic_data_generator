"""User agent definitions and generation."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import numpy as np
from faker import Faker


class IncomeLevel(Enum):
    """User income level categories."""
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8


class TechSavviness(Enum):
    """User tech comfort level."""
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.9


class PrivacySensitivity(Enum):
    """User privacy concern level."""
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.9


@dataclass
class Device:
    """Represents a user device."""
    device_id: str
    device_type: str  # mobile, desktop, tablet, smart_tv
    os: str  # iOS, Android, Windows, macOS, Linux
    created_at: str
    last_used: str = ""
    
    def __post_init__(self):
        if not self.device_id:
            self.device_id = f"device_{uuid.uuid4().hex[:12]}"


@dataclass
class Persona:
    """Represents a user persona template."""
    name: str
    description: str
    income_level: IncomeLevel
    tech_savviness: TechSavviness
    privacy_sensitivity: PrivacySensitivity
    base_search_frequency: float  # searches per day
    ad_click_propensity: float  # 0-1
    impulse_buying: float  # 0-1 multiplier
    device_pref_mobile: float  # prob of choosing mobile
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "income_level": self.income_level.name,
            "tech_savviness": self.tech_savviness.name,
            "privacy_sensitivity": self.privacy_sensitivity.name,
            "base_search_frequency": self.base_search_frequency,
            "ad_click_propensity": self.ad_click_propensity,
            "impulse_buying": self.impulse_buying,
        }


@dataclass
class User:
    """Represents a synthetic user."""
    user_id: str
    persona: Persona
    devices: List[Device] = field(default_factory=list)
    home_sector: str = ""
    purchase_history: List[Dict] = field(default_factory=list)
    contact_network: List[str] = field(default_factory=list)  # other user_ids
    
    # Latent interest fields
    latent_tech_interest: float = 0.5
    latent_finance_interest: float = 0.5
    social_connectivity_score: float = 0.5
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = f"U_{uuid.uuid4().hex[:6].upper()}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "persona": self.persona.to_dict(),
            "home_sector": self.home_sector,
            "latent_tech_interest": self.latent_tech_interest,
            "latent_finance_interest": self.latent_finance_interest,
            "social_connectivity_score": self.social_connectivity_score,
            "n_devices": len(self.devices),
            "n_contacts": len(self.contact_network),
        }


class PersonaFactory:
    """Factory for generating persona templates."""
    
    @staticmethod
    def get_all_personas() -> List[Persona]:
        """Return all persona templates."""
        return [
            PersonaFactory.tech_savvy_millennial(),
            PersonaFactory.budget_conscious_parent(),
            PersonaFactory.business_professional(),
            PersonaFactory.casual_browser(),
            PersonaFactory.privacy_focused_user(),
        ]
    
    @staticmethod
    def tech_savvy_millennial() -> Persona:
        """Tech-savvy young professional."""
        return Persona(
            name="Tech-Savvy Millennial",
            description="Early adopter, high search frequency, mobile-first",
            income_level=IncomeLevel.HIGH,
            tech_savviness=TechSavviness.HIGH,
            privacy_sensitivity=PrivacySensitivity.LOW,
            base_search_frequency=5.0,
            ad_click_propensity=0.4,
            impulse_buying=0.7,
            device_pref_mobile=0.8,
        )
    
    @staticmethod
    def budget_conscious_parent() -> Persona:
        """Price-sensitive family-oriented user."""
        return Persona(
            name="Budget-Conscious Parent",
            description="Deal-hunter, moderate search, family-focused",
            income_level=IncomeLevel.MEDIUM,
            tech_savviness=TechSavviness.MEDIUM,
            privacy_sensitivity=PrivacySensitivity.MEDIUM,
            base_search_frequency=3.0,
            ad_click_propensity=0.3,
            impulse_buying=0.3,
            device_pref_mobile=0.6,
        )
    
    @staticmethod
    def business_professional() -> Persona:
        """Corporate user with high LTV."""
        return Persona(
            name="Business Professional",
            description="High income, B2B focus, desktop-first",
            income_level=IncomeLevel.HIGH,
            tech_savviness=TechSavviness.HIGH,
            privacy_sensitivity=PrivacySensitivity.MEDIUM,
            base_search_frequency=4.0,
            ad_click_propensity=0.5,
            impulse_buying=0.5,
            device_pref_mobile=0.4,
        )
    
    @staticmethod
    def casual_browser() -> Persona:
        """Low-engagement user."""
        return Persona(
            name="Casual Browser",
            description="Passive consumption, low search, low conversion",
            income_level=IncomeLevel.MEDIUM,
            tech_savviness=TechSavviness.LOW,
            privacy_sensitivity=PrivacySensitivity.MEDIUM,
            base_search_frequency=1.0,
            ad_click_propensity=0.1,
            impulse_buying=0.1,
            device_pref_mobile=0.5,
        )
    
    @staticmethod
    def privacy_focused_user() -> Persona:
        """Privacy-conscious user."""
        return Persona(
            name="Privacy-Focused User",
            description="Ad-averse, VPN usage, minimal data sharing",
            income_level=IncomeLevel.MEDIUM,
            tech_savviness=TechSavviness.HIGH,
            privacy_sensitivity=PrivacySensitivity.HIGH,
            base_search_frequency=2.0,
            ad_click_propensity=0.05,
            impulse_buying=0.2,
            device_pref_mobile=0.5,
        )


class UserGenerator:
    """Generates synthetic users with realistic attribute distributions."""
    
    def __init__(self, config: Dict, random_seed: int = None):
        """Initialize user generator.
        
        Args:
            config: Configuration dictionary with simulation params
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.fake = Faker()
        self.personas = PersonaFactory.get_all_personas()
        self.sectors = ["Sector_1A", "Sector_2A", "Sector_3B", "Sector_4C", 
                       "Sector_5D", "Sector_6E", "Sector_7G"]
        
        if random_seed is not None:
            np.random.seed(random_seed)
            Faker.seed(random_seed)
    
    def generate_users(self, n_users: int) -> List[User]:
        """Generate n_users synthetic users.
        
        Args:
            n_users: Number of users to generate
            
        Returns:
            List of User objects
        """
        users = []
        
        # Persona distribution (roughly equal for diversity)
        persona_assignment = np.random.choice(
            self.personas, 
            size=n_users,
            p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        
        for i in range(n_users):
            persona = persona_assignment[i]
            user = self._create_user(persona, i)
            users.append(user)
        
        # Create social network connections
        self._create_contact_networks(users)
        
        return users
    
    def _create_user(self, persona: Persona, user_idx: int) -> User:
        """Create a single user with realistic attributes."""
        user = User(
            user_id=f"U_{user_idx:06d}",
            persona=persona,
            home_sector=np.random.choice(self.sectors),
        )
        
        # Set latent interests
        if persona.tech_savviness == TechSavviness.HIGH:
            user.latent_tech_interest = np.random.beta(5, 2)  # biased high
        else:
            user.latent_tech_interest = np.random.beta(2, 3)  # biased low
        
        user.latent_finance_interest = persona.income_level.value + np.random.normal(0, 0.1)
        user.latent_finance_interest = np.clip(user.latent_finance_interest, 0, 1)
        
        # Social connectivity based on tech savviness
        base_connectivity = persona.tech_savviness.value
        user.social_connectivity_score = np.clip(
            base_connectivity + np.random.normal(0, 0.15),
            0, 1
        )
        
        # Generate devices
        n_devices = self._sample_n_devices(persona)
        for j in range(n_devices):
            device = self._create_device(persona, j)
            user.devices.append(device)
        
        return user
    
    def _sample_n_devices(self, persona: Persona) -> int:
        """Sample number of devices per user based on persona."""
        # Tech-savvy users have more devices
        if persona.tech_savviness == TechSavviness.HIGH:
            return np.random.choice([2, 3, 4], p=[0.5, 0.35, 0.15])
        elif persona.tech_savviness == TechSavviness.MEDIUM:
            return np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        else:
            return np.random.choice([1, 2], p=[0.7, 0.3])
    
    def _create_device(self, persona: Persona, device_idx: int) -> Device:
        """Create a device for a user."""
        # Choose device type based on persona and index
        if device_idx == 0:  # Primary device
            if np.random.random() < persona.device_pref_mobile:
                device_type = "mobile"
                os = np.random.choice(["iOS", "Android"], p=[0.4, 0.6])
            else:
                device_type = "desktop"
                os = np.random.choice(["Windows", "macOS"], p=[0.7, 0.3])
        else:  # Secondary devices
            device_type = np.random.choice(
                ["mobile", "desktop", "tablet"],
                p=[0.5, 0.3, 0.2]
            )
            if device_type == "mobile":
                os = np.random.choice(["iOS", "Android"], p=[0.4, 0.6])
            else:
                os = np.random.choice(["Windows", "macOS"], p=[0.7, 0.3])
        
        device = Device(
            device_id=f"{uuid.uuid4().hex[:12]}",
            device_type=device_type,
            os=os,
            created_at=self.fake.date_time_this_year().isoformat(),
        )
        return device
    
    def _create_contact_networks(self, users: List[User]):
        """Create social connections between users."""
        n_users = len(users)
        
        for user in users:
            # Network size based on social connectivity
            target_contacts = max(1, int(user.social_connectivity_score * 20))
            n_contacts = np.random.poisson(target_contacts)
            n_contacts = min(n_contacts, n_users - 1)
            
            # Sample contacts
            candidates = [u.user_id for u in users if u.user_id != user.user_id]
            contacts = np.random.choice(candidates, size=n_contacts, replace=False)
            user.contact_network = contacts.tolist()
