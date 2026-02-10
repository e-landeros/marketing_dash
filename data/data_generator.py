"""
Synthetic Data Generator for Auto Insurance Marketing Analytics
================================================================
Generates realistic marketing funnel data with embedded patterns for:
- Traffic source performance differences
- Audience segmentation signals
- Fraud/bot detection patterns

Author: Fabian Landeros
"""

import os
import hashlib
import uuid
import warnings
from datetime import datetime, timedelta
from typing import Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Set seed for reproducibility
np.random.seed(42)


# =============================================================================
# CONFIGURATION - Realistic Patterns Encoded Here
# =============================================================================

TRAFFIC_SOURCE_CONFIG = {
    "facebook": {
        "weight": 0.25,  # 25% of traffic
        "age_mean": 35,
        "age_std": 12,
        "mobile_pct": 0.75,
        "cpc_range": (0.80, 2.50),
        "base_conversion_rate": 0.045,
        "fraud_rate": 0.08,
        "campaigns": [
            "fb_retarget_auto",
            "fb_lookalike_q1",
            "fb_broad_auto",
            "fb_video_awareness",
        ],
        "utm_medium": "paid_social",
    },
    "google": {
        "weight": 0.20,
        "age_mean": 42,
        "age_std": 14,
        "mobile_pct": 0.55,
        "cpc_range": (2.00, 6.00),
        "base_conversion_rate": 0.085,  # Higher intent
        "fraud_rate": 0.03,
        "campaigns": [
            "ggl_brand_auto",
            "ggl_competitor_conquest",
            "ggl_generic_insurance",
            "ggl_pmax_auto",
        ],
        "utm_medium": "cpc",
    },
    "tiktok": {
        "weight": 0.18,
        "age_mean": 26,
        "age_std": 6,  # Younger audience
        "mobile_pct": 0.96,
        "cpc_range": (0.30, 1.20),
        "base_conversion_rate": 0.025,  # Lower intent
        "fraud_rate": 0.12,
        "campaigns": ["tt_spark_auto", "tt_ugc_savings", "tt_young_drivers"],
        "utm_medium": "paid_social",
    },
    "taboola": {
        "weight": 0.15,
        "age_mean": 48,
        "age_std": 15,
        "mobile_pct": 0.65,
        "cpc_range": (0.15, 0.60),  # Cheap traffic
        "base_conversion_rate": 0.018,
        "fraud_rate": 0.22,  # High fraud on native
        "campaigns": ["tab_native_auto", "tab_content_discovery"],
        "utm_medium": "native",
    },
    "email": {
        "weight": 0.12,
        "age_mean": 45,
        "age_std": 12,
        "mobile_pct": 0.60,
        "cpc_range": (0.05, 0.15),  # Owned channel
        "base_conversion_rate": 0.12,  # Warm audience
        "fraud_rate": 0.02,
        "campaigns": ["email_renewal_reminder", "email_winback", "email_cross_sell"],
        "utm_medium": "email",
    },
    "affiliate": {
        "weight": 0.10,
        "age_mean": 38,
        "age_std": 14,
        "mobile_pct": 0.70,
        "cpc_range": (1.50, 4.00),
        "base_conversion_rate": 0.055,
        "fraud_rate": 0.18,  # Some shady affiliates
        "campaigns": ["aff_comparison_sites", "aff_cashback", "aff_insurance_blogs"],
        "utm_medium": "affiliate",
    },
}

# Publisher IDs per source (some are "bad actors")
PUBLISHER_CONFIG = {
    "facebook": ["fb_main", "fb_instagram", "fb_audience_network"],
    "google": ["ggl_search", "ggl_display", "ggl_youtube"],
    "tiktok": ["tt_fyp", "tt_search"],
    "taboola": [
        "tab_pub_001",
        "tab_pub_002",
        "tab_pub_003_FRAUD",
        "tab_pub_004",
    ],  # 003 is bad
    "email": ["email_main", "email_partner"],
    "affiliate": ["aff_001", "aff_002_FRAUD", "aff_003", "aff_004"],  # 002 is bad
}

# State distribution with different conversion patterns
STATE_CONFIG = {
    "CA": {"weight": 0.14, "conv_multiplier": 1.1, "payout_multiplier": 1.3},
    "TX": {"weight": 0.12, "conv_multiplier": 1.0, "payout_multiplier": 1.0},
    "FL": {"weight": 0.10, "conv_multiplier": 0.95, "payout_multiplier": 1.1},
    "NY": {"weight": 0.08, "conv_multiplier": 0.9, "payout_multiplier": 1.4},
    "PA": {"weight": 0.05, "conv_multiplier": 1.05, "payout_multiplier": 0.95},
    "IL": {"weight": 0.05, "conv_multiplier": 1.0, "payout_multiplier": 1.0},
    "OH": {"weight": 0.05, "conv_multiplier": 1.1, "payout_multiplier": 0.85},
    "GA": {"weight": 0.04, "conv_multiplier": 1.0, "payout_multiplier": 0.95},
    "NC": {"weight": 0.04, "conv_multiplier": 1.05, "payout_multiplier": 0.9},
    "MI": {"weight": 0.04, "conv_multiplier": 0.95, "payout_multiplier": 1.2},
    "OTHER": {"weight": 0.29, "conv_multiplier": 1.0, "payout_multiplier": 1.0},
}

INSURERS = [
    "geico",
    "progressive",
    "state_farm",
    "allstate",
    "liberty_mutual",
    "usaa",
    "none",
    "other",
]
COVERAGE_TYPES = ["minimum", "liability", "full"]
CREATIVE_VARIANTS = [
    "savings_focus",
    "speed_focus",
    "trust_focus",
    "price_compare",
    "testimonial",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_user_id() -> str:
    """Generate unique user ID"""
    return f"usr_{uuid.uuid4().hex[:10]}"


def generate_timestamp(start_date: datetime, end_date: datetime) -> datetime:
    """Generate random timestamp with realistic hourly distribution"""
    # Bias toward business hours, but include some late night (bot pattern)
    delta = end_date - start_date
    random_days = np.random.randint(0, delta.days)

    # Hour distribution: peaks at 10am, 2pm, 8pm
    hour_weights = [
        0.5,
        0.3,
        0.2,
        0.3,
        0.4,
        0.6,  # 0-5am (low, fraud hours)
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        2.8,  # 6-11am
        2.5,
        2.8,
        3.0,
        2.5,
        2.0,
        2.2,  # 12-5pm
        2.5,
        3.0,
        3.2,
        2.8,
        2.0,
        1.0,  # 6-11pm
    ]
    hour_probs = np.array(hour_weights) / sum(hour_weights)
    hour = np.random.choice(24, p=hour_probs)

    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)

    return start_date + timedelta(
        days=random_days, hours=hour, minutes=minute, seconds=second
    )


def generate_behavioral_signals(is_fraud: bool, device_type: str) -> Dict:
    """Generate behavioral signals with fraud patterns embedded"""

    if is_fraud:
        # BOT PATTERNS
        session_duration = np.random.exponential(5) + 2  # Very short
        pages_viewed = np.random.choice([1, 2], p=[0.7, 0.3])
        form_time = np.random.uniform(3, 12)  # Impossibly fast
        field_corrections = 0  # Bots don't make mistakes
        scroll_depth = np.random.uniform(0, 0.15)  # Barely scrolls
        mouse_score = np.random.uniform(0.1, 0.3)  # Linear, robotic
        clicks = np.random.randint(1, 3)
    else:
        # HUMAN PATTERNS
        if device_type == "mobile":
            session_duration = np.random.lognormal(4.5, 0.8)  # ~90 sec median
            form_time = np.random.lognormal(4.2, 0.5)  # ~65 sec median
        else:
            session_duration = np.random.lognormal(5.0, 0.7)  # ~150 sec median
            form_time = np.random.lognormal(4.0, 0.4)  # ~55 sec median

        pages_viewed = np.random.poisson(3) + 1
        field_corrections = np.random.poisson(2)
        scroll_depth = np.random.beta(5, 2)  # Most scroll to ~70%
        mouse_score = np.random.beta(6, 2)  # Natural, erratic movements
        clicks = np.random.poisson(5) + 2

    return {
        "session_duration_sec": max(2, min(session_duration, 600)),
        "pages_viewed": max(1, min(pages_viewed, 15)),
        "form_start_to_submit_sec": max(3, min(form_time, 300)),
        "field_corrections": min(field_corrections, 15),
        "scroll_depth_pct": round(scroll_depth, 3),
        "mouse_movement_score": round(mouse_score, 3),
        "clicks_before_submit": max(1, min(clicks, 25)),
    }


def generate_user_demographics(source_config: Dict, is_fraud: bool) -> Dict:
    """Generate user demographics based on source characteristics"""

    if is_fraud:
        # Fraud profiles often have synthetic/random data
        age = np.random.randint(18, 65)
        gender = np.random.choice(["M", "F", "Unknown"], p=[0.4, 0.4, 0.2])
    else:
        age = int(
            np.clip(
                np.random.normal(source_config["age_mean"], source_config["age_std"]),
                18,
                80,
            )
        )
        gender = np.random.choice(["M", "F", "Unknown"], p=[0.48, 0.47, 0.05])

    # Driving experience correlated with age
    min_years = max(0, age - 18)
    years_driving = np.random.randint(0, min(min_years + 1, 50))

    # Vehicles correlated with age
    if age < 25:
        vehicles = np.random.choice([1, 2], p=[0.85, 0.15])
    elif age < 45:
        vehicles = np.random.choice([1, 2, 3], p=[0.4, 0.45, 0.15])
    else:
        vehicles = np.random.choice([1, 2, 3, 4], p=[0.35, 0.4, 0.2, 0.05])

    # Accidents - younger drivers have more
    if age < 25:
        accidents = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
    else:
        accidents = np.random.choice([0, 1, 2], p=[0.75, 0.2, 0.05])

    # Coverage preference by age
    if age < 30:
        coverage = np.random.choice(COVERAGE_TYPES, p=[0.35, 0.45, 0.20])
    elif age < 50:
        coverage = np.random.choice(COVERAGE_TYPES, p=[0.15, 0.40, 0.45])
    else:
        coverage = np.random.choice(COVERAGE_TYPES, p=[0.10, 0.35, 0.55])

    return {
        "age": age,
        "gender": gender,
        "vehicles_count": vehicles,
        "years_driving": years_driving,
        "accidents_last_3yr": accidents,
        "current_insurer": np.random.choice(INSURERS),
        "coverage_type": coverage,
    }


def calculate_conversion_probability(
    base_rate: float,
    state_multiplier: float,
    age: int,
    coverage_type: str,
    device_type: str,
    hour: int,
    is_fraud: bool,
) -> float:
    """Calculate conversion probability with multiple factors"""

    if is_fraud:
        # Fraud rarely converts at final stage
        return base_rate * 0.05

    prob = base_rate * state_multiplier

    # Age factor (prime age 35-55)
    if 35 <= age <= 55:
        prob *= 1.2
    elif age < 25:
        prob *= 0.7

    # Coverage type (full coverage = serious buyer)
    if coverage_type == "full":
        prob *= 1.3
    elif coverage_type == "minimum":
        prob *= 0.8

    # Device (desktop converts better)
    if device_type == "desktop":
        prob *= 1.15
    elif device_type == "tablet":
        prob *= 1.05

    # Time of day (business hours convert better)
    if 9 <= hour <= 17:
        prob *= 1.1
    elif hour < 6 or hour > 22:
        prob *= 0.7

    return min(prob, 0.25)  # Cap at 25%


def generate_funnel_events(conv_prob: float, is_fraud: bool) -> Dict:
    """Generate funnel progression with realistic drop-off"""

    landed = True  # Everyone lands

    # Fraud has weird funnel behavior
    if is_fraud:
        started_quote = np.random.random() < 0.9  # Bots usually start form
        completed_quote = started_quote and (np.random.random() < 0.85)
        submitted = completed_quote and (np.random.random() < 0.7)
        accepted = submitted and (np.random.random() < 0.1)  # Carriers reject fraud
        sold = accepted and (np.random.random() < 0.05)
    else:
        # Normal funnel with realistic drop-off
        started_quote = np.random.random() < 0.65
        completed_quote = started_quote and (np.random.random() < 0.55)
        submitted = completed_quote and (np.random.random() < 0.80)

        # Final conversion based on calculated probability
        if submitted:
            accepted = np.random.random() < 0.60
            sold = accepted and (np.random.random() < (conv_prob / 0.30))  # Normalized
        else:
            accepted = False
            sold = False

    return {
        "landed": landed,
        "started_quote": started_quote,
        "completed_quote": completed_quote,
        "submitted_to_carrier": submitted,
        "carrier_accepted": accepted,
        "policy_sold": sold,
    }


def calculate_payout(
    sold: bool, state_config: Dict, coverage_type: str, age: int
) -> float:
    """Calculate payout amount for sold policies"""

    if not sold:
        return 0.0

    # Base payout
    base_payout = np.random.uniform(40, 80)

    # State multiplier
    payout = base_payout * state_config["payout_multiplier"]

    # Coverage type bonus
    if coverage_type == "full":
        payout *= 1.4
    elif coverage_type == "liability":
        payout *= 1.1

    # Age factor (middle age pays more)
    if 30 <= age <= 60:
        payout *= 1.15

    return round(payout, 2)


# =============================================================================
# MAIN GENERATOR
# =============================================================================


def generate_dataset(
    n_records: int = 200000,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Generate synthetic marketing dataset with embedded patterns.

    Parameters:
    -----------
    n_records : int
        Number of records to generate
    start_date : str
        Start date for timestamp range
    end_date : str
        End date for timestamp range

    Returns:
    --------
    pd.DataFrame
        Complete synthetic dataset
    """

    print(f"Generating {n_records:,} synthetic records...")
    print("=" * 50)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Pre-calculate source weights
    sources = list(TRAFFIC_SOURCE_CONFIG.keys())
    source_weights = [TRAFFIC_SOURCE_CONFIG[s]["weight"] for s in sources]

    # Pre-calculate state weights
    states = list(STATE_CONFIG.keys())
    state_weights = [STATE_CONFIG[s]["weight"] for s in states]

    records = []

    for i in range(n_records):
        if (i + 1) % 50000 == 0:
            print(f"  Generated {i + 1:,} records...")

        # Select traffic source
        source = np.random.choice(sources, p=source_weights)
        source_config = TRAFFIC_SOURCE_CONFIG[source]

        # Select state
        state = np.random.choice(states, p=state_weights)
        state_config = STATE_CONFIG[state]

        # Determine if this is a fraud record
        publisher_options = PUBLISHER_CONFIG[source]
        publisher = np.random.choice(publisher_options)

        # Publishers with "FRAUD" in name have 60% fraud rate, others use source rate
        if "FRAUD" in publisher:
            is_fraud = np.random.random() < 0.60
        else:
            is_fraud = np.random.random() < source_config["fraud_rate"]

        # Generate timestamp
        timestamp = generate_timestamp(start_dt, end_dt)
        hour = timestamp.hour

        # Device selection
        if np.random.random() < source_config["mobile_pct"]:
            device_type = "mobile"
            os_choices = ["iOS", "Android"]
            os_weights = [0.45, 0.55]
        else:
            device_type = np.random.choice(["desktop", "tablet"], p=[0.85, 0.15])
            if device_type == "desktop":
                os_choices = ["Windows", "macOS", "Linux"]
                os_weights = [0.70, 0.25, 0.05]
            else:
                os_choices = ["iOS", "Android"]
                os_weights = [0.60, 0.40]

        os_system = np.random.choice(os_choices, p=os_weights)

        # Browser based on OS
        if os_system in ["iOS", "macOS"]:
            browser = np.random.choice(["Safari", "Chrome"], p=[0.65, 0.35])
        elif os_system == "Android":
            browser = np.random.choice(["Chrome", "Samsung Browser"], p=[0.80, 0.20])
        else:
            browser = np.random.choice(
                ["Chrome", "Firefox", "Edge"], p=[0.65, 0.20, 0.15]
            )

        # Screen resolution
        if device_type == "mobile":
            resolution = np.random.choice(["375x812", "390x844", "412x915", "360x800"])
        elif device_type == "tablet":
            resolution = np.random.choice(["768x1024", "810x1080", "820x1180"])
        else:
            resolution = np.random.choice(
                ["1920x1080", "1366x768", "2560x1440", "1536x864"]
            )

        # Fraud signals
        is_proxy = (
            is_fraud
            and (np.random.random() < 0.4)
            or (not is_fraud and np.random.random() < 0.02)
        )
        is_datacenter = (
            is_fraud
            and (np.random.random() < 0.35)
            or (not is_fraud and np.random.random() < 0.005)
        )

        # Generate behavioral and demographic data
        behavior = generate_behavioral_signals(is_fraud, device_type)
        demographics = generate_user_demographics(source_config, is_fraud)

        # Calculate conversion probability
        conv_prob = calculate_conversion_probability(
            base_rate=source_config["base_conversion_rate"],
            state_multiplier=state_config["conv_multiplier"],
            age=demographics["age"],
            coverage_type=demographics["coverage_type"],
            device_type=device_type,
            hour=hour,
            is_fraud=is_fraud,
        )

        # Generate funnel events
        funnel = generate_funnel_events(conv_prob, is_fraud)

        # Calculate payout
        payout = calculate_payout(
            sold=funnel["policy_sold"],
            state_config=state_config,
            coverage_type=demographics["coverage_type"],
            age=demographics["age"],
        )

        # CPC
        cpc = round(np.random.uniform(*source_config["cpc_range"]), 2)

        # Build record
        record = {
            # IDs
            "user_id": generate_user_id(),
            "timestamp": timestamp,
            # Attribution
            "traffic_source": source,
            "campaign_id": np.random.choice(source_config["campaigns"]),
            "publisher_id": publisher,
            "utm_source": source,
            "utm_medium": source_config["utm_medium"],
            "creative_id": np.random.choice(CREATIVE_VARIANTS),
            "cost_per_click": cpc,
            # Device & Technical
            "device_type": device_type,
            "os": os_system,
            "browser": browser,
            "screen_resolution": resolution,
            "state": state,
            "is_proxy": is_proxy,
            "is_datacenter_ip": is_datacenter,
            "user_agent_hash": f"ua_{hashlib.md5(f'{browser}{os_system}{resolution}'.encode()).hexdigest()[:8]}",
            # Behavioral
            **behavior,
            # Time features
            "hour_of_day": hour,
            "day_of_week": timestamp.weekday(),
            # Demographics
            **demographics,
            "zip_code": f"{np.random.randint(10000, 99999)}",
            # Funnel
            **funnel,
            # Revenue
            "payout_amount": payout,
            # Ground truth (for validation, would not exist in real data)
            "_is_fraud_synthetic": is_fraud,
        }

        records.append(record)

    df = pd.DataFrame(records)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("=" * 50)
    print("Generation complete!")
    print(f"\nDataset Summary:")
    print(f"  Total records: {len(df):,}")
    print(
        f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    )
    print(f"  Fraud rate: {df['_is_fraud_synthetic'].mean():.1%}")
    print(f"  Overall conversion rate: {df['policy_sold'].mean():.2%}")
    print(f"  Total revenue: ${df['payout_amount'].sum():,.2f}")

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print detailed summary statistics"""

    print("\n" + "=" * 60)
    print("DETAILED DATASET SUMMARY")
    print("=" * 60)

    # By source
    print("\n📊 Performance by Traffic Source:")
    print("-" * 50)
    source_stats = (
        df.groupby("traffic_source")
        .agg(
            {
                "user_id": "count",
                "policy_sold": "mean",
                "payout_amount": "sum",
                "cost_per_click": "mean",
                "_is_fraud_synthetic": "mean",
            }
        )
        .round(4)
    )
    source_stats.columns = ["Volume", "Conv Rate", "Revenue", "Avg CPC", "Fraud Rate"]
    source_stats["ROAS"] = (
        source_stats["Revenue"] / (source_stats["Volume"] * source_stats["Avg CPC"])
    ).round(2)
    print(source_stats.sort_values("Revenue", ascending=False).to_string())

    # By device
    print("\n📱 Performance by Device:")
    print("-" * 50)
    device_stats = (
        df.groupby("device_type")
        .agg({"user_id": "count", "policy_sold": "mean", "_is_fraud_synthetic": "mean"})
        .round(4)
    )
    device_stats.columns = ["Volume", "Conv Rate", "Fraud Rate"]
    print(device_stats.to_string())

    # Funnel analysis
    print("\n🔄 Funnel Conversion Rates:")
    print("-" * 50)
    funnel_cols = [
        "landed",
        "started_quote",
        "completed_quote",
        "submitted_to_carrier",
        "carrier_accepted",
        "policy_sold",
    ]
    for col in funnel_cols:
        rate = df[col].mean()
        print(f"  {col:25s}: {rate:7.2%}")

    # Fraud patterns
    print("\n🚨 Fraud Patterns by Publisher (Top 10):")
    print("-" * 50)
    pub_fraud = (
        df.groupby("publisher_id")
        .agg({"user_id": "count", "_is_fraud_synthetic": "mean"})
        .sort_values("_is_fraud_synthetic", ascending=False)
        .head(10)
    )
    pub_fraud.columns = ["Volume", "Fraud Rate"]
    print(pub_fraud.to_string())


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Generate dataset
    df = generate_dataset(n_records=200000)

    # Print summary
    print_dataset_summary(df)

    # Save to CSV - use path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "data", "raw", "synthetic_leads.csv")

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")
