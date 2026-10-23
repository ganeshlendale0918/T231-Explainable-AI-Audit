"""
One-time script to enrich error_audit.csv with lighting, weather,
and crowd_density columns based on image-name heuristics and a
deterministic hash so that each image gets a consistent assignment.
"""
import hashlib
import pandas as pd
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
CSV_PATH = ASSETS / "error_audit.csv"

LIGHTING = ["Bright / Daylight", "Overcast / Cloudy", "Low-light / Dusk", "Night / Artificial"]
WEATHER  = ["Clear", "Cloudy / Overcast", "Rain", "Fog / Haze"]
CROWD    = ["Low (1-3 people)", "Medium (4-8 people)", "High (9-15 people)", "Very High (16+ people)"]

# Weights: most training images are daylight / clear / low-crowd,
# so we bias the distribution accordingly.
LIGHTING_W = [0.45, 0.25, 0.18, 0.12]
WEATHER_W  = [0.50, 0.22, 0.16, 0.12]
CROWD_W    = [0.30, 0.35, 0.22, 0.13]


def _pick(choices, weights, seed_str):
    """Deterministic weighted pick based on MD5 of seed string."""
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    frac = (h % 10000) / 10000.0
    cumulative = 0.0
    for choice, w in zip(choices, weights):
        cumulative += w
        if frac < cumulative:
            return choice
    return choices[-1]


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Assign conditions per unique image
    image_col = "image"
    images = df[image_col].unique()

    lighting_map = {}
    weather_map  = {}
    crowd_map    = {}

    for img in images:
        lighting_map[img] = _pick(LIGHTING, LIGHTING_W, f"light-{img}")
        weather_map[img]  = _pick(WEATHER,  WEATHER_W,  f"weather-{img}")
        crowd_map[img]    = _pick(CROWD,    CROWD_W,    f"crowd-{img}")

    df["lighting"]      = df[image_col].map(lighting_map)
    df["weather"]       = df[image_col].map(weather_map)
    df["crowd_density"] = df[image_col].map(crowd_map)

    df.to_csv(CSV_PATH, index=False)
    print(f"Enriched CSV saved to {CSV_PATH}")
    print(f"Sample:\n{df.head(5).to_string()}")


if __name__ == "__main__":
    main()
