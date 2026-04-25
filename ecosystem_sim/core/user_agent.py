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

    def __init__(
        self, config: Dict, random_seed: int = None, priors: Optional[Dict] = None
    ):
        """Initialize user generator.

        Args:
            config: Configuration dictionary with simulation params
            random_seed: Random seed for reproducibility
            priors: Optional compiled dataset priors
        """
        self.config = config
        self.fake = Faker()
        self.priors = priors

        # Load data-driven personas from priors
        self.personas = self._load_data_driven_personas()

        # Load sectors from Urban/Atlas priors if available
        self.sectors = self._load_sectors_from_priors()

        self._os_mobile = self._load_os_split("mobile")
        self._os_desktop = self._load_os_split("desktop")

        if random_seed is not None:
            np.random.seed(random_seed)
            Faker.seed(random_seed)

    def _map_to_enum(self, value: float, enum_class) -> Enum:
        """Map a float value (0-1) to the nearest enum level."""
        if value > 0.7:
            # Get the HIGH variant of the enum
            for member in enum_class:
                if member.value >= 0.8:
                    return member
        elif value < 0.3:
            for member in enum_class:
                if member.value <= 0.2:
                    return member
        else:
            for member in enum_class:
                if 0.4 <= member.value <= 0.6:
                    return member
        # Fallback: return the first member
        return list(enum_class)[0]

    def _derive_personas_from_priors(self, n_personas: int) -> List[Persona]:
        """Generate N personas by sampling from prior distributions.

        Uses income, device OS, developer activity, and other priors
        to create diverse, data-driven persona templates.

        Args:
            n_personas: Number of personas to generate

        Returns:
            List of Persona objects
        """
        from .prior_loader import get_prior

        # Get distributions from priors
        income_dist = get_prior(
            self.priors, "layer_1_persona.income_distribution.fit_params", None
        )
        device_os = get_prior(
            self.priors, "layer_1_persona.device_os_distribution", {}
        )
        dev_activity = get_prior(
            self.priors,
            "layer_1_persona.developer_activity_levels.developer_types",
            None,
        )
        uplift_conversion = get_prior(
            self.priors,
            "layer_3_intelligence.propensity.conversion_rate_by_persona",
            None,
        )

        personas = []
        for i in range(n_personas):
            # Sample income from lognormal distribution
            if income_dist:
                mu = income_dist.get("mu", 10.8)
                sigma = income_dist.get("sigma", 0.95)
                income_raw = float(np.random.lognormal(mu, sigma))
                income_level_val = min(1.0, income_raw / 200000.0)
            else:
                income_level_val = np.random.uniform(0.2, 0.9)

            # Tech savviness correlates with mobile OS preference
            mobile_ios = device_os.get("mobile_ios", 0.38)
            tech_val = float(
                np.clip(np.random.beta(2 + mobile_ios * 10, 2 + (1 - mobile_ios) * 10), 0.1, 0.95)
            )

            # Privacy sensitivity (inverse of social connectivity)
            privacy_val = float(np.random.beta(2, 5))

            # Search frequency from power-law distribution
            search_freq = float(max(0.5, np.random.pareto(2.5) + 0.5))

            # Ad click propensity (beta distribution centered on 0.2)
            ad_click = float(np.random.beta(2, 8))

            # Impulse buying (correlated with income)
            impulse = float(
                np.clip(np.random.beta(2 + income_level_val * 3, 5 - income_level_val * 2), 0.05, 0.95)
            )

            # Device preference (slightly mobile-biased)
            device_pref = float(np.random.beta(3, 2))

            # Optionally use developer activity distribution for variety
            if dev_activity and isinstance(dev_activity, dict):
                dev_types = list(dev_activity.keys())
                dev_probs = list(dev_activity.values())
                if dev_probs and sum(dev_probs) > 0:
                    chosen_dev_type = np.random.choice(dev_types, p=dev_probs)
                    # Developer-active personas have higher tech savviness
                    if chosen_dev_type in ("active", "core"):
                        tech_val = min(0.95, tech_val + 0.2)
                        search_freq = min(10.0, search_freq + 2.0)

            persona = Persona(
                name=f"Data-Driven Persona {i + 1}",
                description=(
                    f"Derived from prior distributions (cluster {i + 1}): "
                    f"income={income_level_val:.2f}, tech={tech_val:.2f}"
                ),
                income_level=self._map_to_enum(income_level_val, IncomeLevel),
                tech_savviness=self._map_to_enum(tech_val, TechSavviness),
                privacy_sensitivity=self._map_to_enum(privacy_val, PrivacySensitivity),
                base_search_frequency=min(search_freq, 10.0),
                ad_click_propensity=min(ad_click, 1.0),
                impulse_buying=min(impulse, 1.0),
                device_pref_mobile=min(device_pref, 1.0),
            )
            personas.append(persona)

        return personas

    def _load_data_driven_personas(self) -> List[Persona]:
        """Load persona templates from dataset priors.

        Priority order:
        1. Pre-computed persona clusters from priors (if available)
        2. Derive personas by sampling from prior distributions
        3. Fall back to static PersonaFactory templates
        """
        from .prior_loader import get_prior

        n_personas = self.config.get("persona", {}).get("n_personas", 5)

        # Priority 1: Try pre-computed persona clusters from priors
        persona_clusters = get_prior(
            self.priors, "layer_1_persona.persona_clusters", None
        )

        if (
            persona_clusters
            and isinstance(persona_clusters, list)
            and len(persona_clusters) > 0
        ):
            # Use data-driven personas from priors
            personas = []
            for i, cluster in enumerate(persona_clusters[:n_personas]):
                name = cluster.get("name", f"Persona_{i + 1}")
                description = cluster.get("description", f"Data-driven persona {i + 1}")

                # Map cluster parameters to persona attributes
                income_val = cluster.get("income_level", 0.5)
                tech_val = cluster.get("tech_savviness", 0.5)
                privacy_val = cluster.get("privacy_sensitivity", 0.5)

                income_level = (
                    IncomeLevel.HIGH
                    if income_val > 0.7
                    else (IncomeLevel.LOW if income_val < 0.3 else IncomeLevel.MEDIUM)
                )
                tech_savviness = (
                    TechSavviness.HIGH
                    if tech_val > 0.7
                    else (TechSavviness.LOW if tech_val < 0.3 else TechSavviness.MEDIUM)
                )
                privacy_sensitivity = (
                    PrivacySensitivity.HIGH
                    if privacy_val > 0.7
                    else (
                        PrivacySensitivity.LOW
                        if privacy_val < 0.3
                        else PrivacySensitivity.MEDIUM
                    )
                )

                persona = Persona(
                    name=name,
                    description=description,
                    income_level=income_level,
                    tech_savviness=tech_savviness,
                    privacy_sensitivity=privacy_sensitivity,
                    base_search_frequency=cluster.get("search_frequency", 3.0),
                    ad_click_propensity=cluster.get("ad_click_propensity", 0.3),
                    impulse_buying=cluster.get("impulse_buying", 0.3),
                    device_pref_mobile=cluster.get("device_pref_mobile", 0.6),
                )
                personas.append(persona)

            if personas:
                return personas

        # Priority 2: Derive personas by sampling from prior distributions
        if self.priors is not None:
            return self._derive_personas_from_priors(n_personas)

        # Priority 3: Fall back to static PersonaFactory templates
        return PersonaFactory.get_all_personas()[:n_personas]

    def _load_sectors_from_priors(self) -> List[str]:
        """Load sectors from Urban/Atlas priors."""
        from .prior_loader import get_prior

        n_sectors = self.config.get("persona", {}).get("n_sectors", 7)

        sector_typology = get_prior(
            self.priors, "layer_1_persona.sector_typology", None
        )
        if sector_typology and isinstance(sector_typology, dict):
            # Use top sectors by prevalence
            sorted_sectors = sorted(
                sector_typology.items(), key=lambda x: x[1], reverse=True
            )
            return [sector for sector, _ in sorted_sectors[:n_sectors]]

        # Default sectors
        return [f"Sector_{i}A" for i in range(1, n_sectors + 1)]

    def _load_os_split(self, device_type: str):
        from .prior_loader import get_prior

        if device_type == "mobile":
            ios = get_prior(
                self.priors, "layer_1_persona.device_os_distribution.mobile_ios", 0.4
            )
            android = 1.0 - ios
            return (["iOS", "Android"], [ios, android])
        else:
            win = get_prior(
                self.priors,
                "layer_1_persona.device_os_distribution.desktop_windows",
                0.7,
            )
            mac = 1.0 - win
            return (["Windows", "macOS"], [win, mac])

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

        user.latent_finance_interest = persona.income_level.value + np.random.normal(
            0, 0.1
        )
        user.latent_finance_interest = np.clip(user.latent_finance_interest, 0, 1)

        # Social connectivity based on tech savviness
        base_connectivity = persona.tech_savviness.value
        user.social_connectivity_score = np.clip(
            base_connectivity + np.random.normal(0, 0.15), 0, 1
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
                os = np.random.choice(self._os_mobile[0], p=self._os_mobile[1])
            else:
                device_type = "desktop"
                os = np.random.choice(self._os_desktop[0], p=self._os_desktop[1])
        else:  # Secondary devices
            device_type = np.random.choice(
                ["mobile", "desktop", "tablet"], p=[0.5, 0.3, 0.2]
            )
            if device_type == "mobile":
                os = np.random.choice(self._os_mobile[0], p=self._os_mobile[1])
            else:
                os = np.random.choice(self._os_desktop[0], p=self._os_desktop[1])

        device = Device(
            device_id=f"{uuid.uuid4().hex[:12]}",
            device_type=device_type,
            os=os,
            created_at=self.fake.date_time_this_year().isoformat(),
        )
        return device

    def generate_users(self, n_users: int) -> List[User]:
        """Generate n_users synthetic users.

        Args:
            n_users: Number of users to generate

        Returns:
            List of User objects
        """
        users = []

        # Get persona distribution from priors if available
        from .prior_loader import get_prior

        persona_dist = get_prior(
            self.priors, "layer_1_persona.persona_distribution", None
        )

        if (
            persona_dist
            and isinstance(persona_dist, list)
            and len(persona_dist) == len(self.personas)
        ):
            # Use data-driven persona distribution
            persona_probs = persona_dist
        else:
            # Equal distribution as fallback
            persona_probs = [1.0 / len(self.personas)] * len(self.personas)

        # Normalize probabilities
        total = sum(persona_probs)
        persona_probs = [p / total for p in persona_probs]

        persona_assignment = np.random.choice(
            self.personas, size=n_users, p=persona_probs
        )

        for i in range(n_users):
            persona = persona_assignment[i]
            user = self._create_user(persona, i)
            users.append(user)

        # Create social network connections using SNAP patterns
        self._create_contact_networks(users)

        return users

    def _create_contact_networks(self, users: List[User]):
        """Create social connections between users."""
        from .prior_loader import get_prior

        n_users = len(users)

        alpha = get_prior(
            self.priors, "layer_1_persona.contact_network_degree.fit_params.alpha", None
        )
        x_min = get_prior(
            self.priors, "layer_1_persona.contact_network_degree.fit_params.x_min", 1.0
        )
        max_degree = get_prior(
            self.priors, "layer_1_persona.contact_network_degree.max_degree", 500
        )

        for user in users:
            if alpha is not None:
                raw = (np.random.pareto(alpha) + 1) * x_min
                target_contacts = int(
                    np.clip(raw * user.social_connectivity_score, 1, max_degree)
                )
            else:
                target_contacts = max(1, int(user.social_connectivity_score * 20))
                target_contacts = np.random.poisson(target_contacts)

            n_contacts = min(target_contacts, n_users - 1)

            candidates = [u.user_id for u in users if u.user_id != user.user_id]
            contacts = np.random.choice(candidates, size=n_contacts, replace=False)
            user.contact_network = contacts.tolist()
