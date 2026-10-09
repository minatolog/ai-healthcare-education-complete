"""Central configuration for the KiTS19 2D segmentation project.

All paths are anchored at PROJECT_ROOT so the code runs identically on the
GPU server (/ssd4/linpf7/kits_project) regardless of the caller's CWD.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw KiTS19 (git repo with data/case_XXXXX/{imaging,segmentation}.nii.gz)
KITS_DATA_DIR = PROJECT_ROOT / "kits19" / "data"

# Derived 2D slice dataset
PROC_DIR = PROJECT_ROOT / "data2d"
IMG_DIR = PROC_DIR / "images"
MASK_DIR = PROC_DIR / "masks"
MANIFEST_CSV = PROC_DIR / "manifest.csv"
SPLIT_JSON = PROC_DIR / "splits.json"
STATS_JSON = PROC_DIR / "dataset_stats.json"

# Outputs
OUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"
FIG_DIR = OUT_DIR / "figures"
RESULT_DIR = OUT_DIR / "results"

# Label map (KiTS19)
LABELS = {0: "background", 1: "kidney", 2: "tumor"}
NUM_CLASSES = 3

# Preprocessing parameters
# Soft-tissue HU window (approx. 0.5/99.5 percentiles used by nnU-Net for KiTS)
HU_MIN = -79.0
HU_MAX = 304.0
SLICE_SIZE = 256           # in-plane resize target (H = W = SLICE_SIZE)
# Fraction of purely-background slices to keep per case (context / true negatives)
NEG_SLICE_FRAC = 0.15

# Split ratios (by CASE to avoid slice leakage)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def ensure_dirs():
    for d in [PROC_DIR, IMG_DIR, MASK_DIR, OUT_DIR, CKPT_DIR, LOG_DIR, FIG_DIR, RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
