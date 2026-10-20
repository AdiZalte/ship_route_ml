
from pathlib import Path
import json
import numpy as np

data_path = Path("data/processed/trajectory_sequences.npz")
report_path = Path("data/processed/sequence_report.json")

if not data_path.exists():
    raise FileNotFoundError(
        "Run python -m src.data.build_sequences first."
    )

with np.load(data_path) as data:
    for name in data.files:
        arr = data[name]
        print(
            name,
            "shape =", arr.shape,
            "dtype =", arr.dtype,
            "finite =", bool(np.isfinite(arr).all())
        )

    if len(data["X_train"]):
        print("Example input positions:")
        print(data["X_train"][0, :, :2])

        print("Example future positions:")
        print(data["y_train"][0])

with open(report_path) as f:
    report = json.load(f)

print("\nSequence report:")
print(json.dumps(report, indent=2))
