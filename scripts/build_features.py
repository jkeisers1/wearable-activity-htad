from pathlib import Path
import sys

# Project structure
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from wearable_htad.features.pipeline import build_full_feature_dataset

raw_folder = project_root / "data" / "htad" / "raw"
output_path = project_root / "data" / "processed" / "features.csv"

# Parameters
sampling_rate = 1000 / 35  # median dt_ms in ms
window_duration = 3.0
window_size = int(sampling_rate * window_duration)
overlap = 0.5

# Run pipeline
print(f"📥 Loading data from: {raw_folder}")
df = build_full_feature_dataset(raw_folder, window_size, overlap)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Saved {len(df)} windows to: {output_path}")
print(f"🔍 Activities: {df['activity'].nunique()}, Users: {df['user'].nunique()}")
