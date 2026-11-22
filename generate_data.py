"""Regenerate the synthetic campus occupancy dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "campus_crowd.csv"
rng = np.random.default_rng(42)
dates = pd.date_range("2026-01-01", periods=180, freq="D")
rows = []

for date in dates:
    dow = date.dayofweek
    weekend = int(dow >= 5)
    day_num = date.dayofyear
    exam = int(day_num in range(35, 49) or day_num in range(115, 129) or day_num in range(155, 169))
    event = int(rng.random() < (0.12 if not weekend else 0.05))
    for hour in range(8, 23):
        for location in ["Library", "Cafeteria"]:
            capacity = 300 if location == "Library" else 180
            if location == "Library":
                level = 28 + 105 * np.exp(-0.5 * ((hour - 16) / 3.0) ** 2)
                weekend_factor = 0.68 if weekend else 1.0
                level += 62 if exam else 0
                level += -8 if event else 0
                noise_sd = 9
            else:
                level = 12 + 78 * np.exp(-0.5 * ((hour - 13) / 1.35) ** 2)
                level += 48 * np.exp(-0.5 * ((hour - 19) / 1.6) ** 2)
                weekend_factor = 0.82 if weekend else 1.0
                level += 5 if exam else 0
                level += 24 if event else 0
                noise_sd = 7
            day_effect = [0, 3, 5, 4, 2, -7, -10][dow]
            count = int(np.clip(round((level + day_effect) * weekend_factor + rng.normal(0, noise_sd)), 0, capacity))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_of_week": dow,
                "hour": hour,
                "is_weekend": weekend,
                "exam_period": exam,
                "campus_event": event,
                "location": location,
                "capacity": capacity,
                "occupancy_count": count,
                "occupancy_pct": round(100 * count / capacity, 1),
            })

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Saved {len(rows):,} rows to {OUT}")
