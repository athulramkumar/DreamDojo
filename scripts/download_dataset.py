"""Download GR1 eval datasets with automatic rate limit retry."""
import time
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

DATASETS = [
    ("In-lab_Eval", ["In-lab_Eval/*"]),
    ("EgoDex_Eval", ["EgoDex_Eval/*"]),
    ("DreamDojo-HV_Eval", ["DreamDojo-HV_Eval/*"]),
]

for name, patterns in DATASETS:
    print(f"\n=== Downloading {name} ===")
    while True:
        try:
            path = snapshot_download(
                "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1",
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1",
            )
            print(f"{name} downloaded to: {path}")
            break
        except HfHubHTTPError as e:
            if "429" in str(e):
                print(f"Rate limited. Waiting 5 minutes before retry...")
                time.sleep(310)
            else:
                raise
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limited. Waiting 5 minutes before retry...")
                time.sleep(310)
            else:
                raise

print("\n=== All eval datasets downloaded ===")
