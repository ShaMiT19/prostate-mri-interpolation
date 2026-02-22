import os
import glob
import pickle
import numpy as np
import pydicom
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# HIPERGATOR PATHS
# ======================================================
BASE_DIR = "/blue/pinaki.sarder/s.savant/prostate_sr_project"
KAGGLE_ROOT = f"{BASE_DIR}/data/raw/Prostate-MRI-US-Biopsy"
OUTPUT_DIR = f"{BASE_DIR}/data/processed"
SPLIT_FILE = f"{OUTPUT_DIR}/splits.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_t2(ds):
    mod = ds.get("Modality", "").upper()
    desc = str(ds.get("SeriesDescription", "")).lower()
    return mod == "MR" and "t2" in desc


def is_axial(ds):
    desc = str(ds.get("SeriesDescription", "")).lower()
    return any(k in desc for k in ["ax", "axial", "axl"])


def load_sorted_slices(files):
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            pos = ds.get("ImagePositionPatient", None)
            z = pos[-1] if pos is not None else ds.get("InstanceNumber", 0)
            slices.append((z, ds))
        except:
            continue

    if not slices:
        return None

    slices.sort(key=lambda x: x[0])
    return np.stack([s[1].pixel_array.astype(np.float32) for s in slices])


def robust_normalize(vol):
    p1 = np.percentile(vol, 1)
    p99 = np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    return (vol - p1) / (p99 - p1 + 1e-8)


def resize_slice(img):
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = TF.resize(t, (256, 256), antialias=True)
    return t[0, 0].numpy()


def resize_volume(vol):
    return np.stack([resize_slice(s) for s in vol])


patients = sorted(os.listdir(KAGGLE_ROOT))

processed, skipped = [], []

for patient in tqdm(patients, desc="Processing"):
    pdir = os.path.join(KAGGLE_ROOT, patient)
    if not os.path.isdir(pdir):
        continue

    candidate = []

    for root, dirs, files in os.walk(pdir):
        dicoms = glob.glob(os.path.join(root, "*.dcm"))
        if not dicoms:
            continue

        try:
            ds = pydicom.dcmread(dicoms[0], stop_before_pixels=True)
        except:
            continue

        if not is_t2(ds):
            continue
        if not is_axial(ds):
            continue

        candidate.append((len(dicoms), dicoms))

    if not candidate:
        skipped.append(patient)
        continue

    candidate.sort(key=lambda x: -x[0])
    _, best_dicoms = candidate[0]

    if len(best_dicoms) < 20:
        skipped.append(patient)
        continue

    vol = load_sorted_slices(best_dicoms)
    if vol is None:
        skipped.append(patient)
        continue

    vol = robust_normalize(vol)
    vol = resize_volume(vol)

    np.save(os.path.join(OUTPUT_DIR, f"{patient}.npy"), vol.astype(np.float32))
    processed.append(patient)

processed = sorted(processed)
np.random.shuffle(processed)

n = len(processed)
splits = {
    "train": processed[: int(0.7*n)],
    "val": processed[int(0.7*n): int(0.85*n)],
    "test": processed[int(0.85*n):],
    "skipped": skipped
}

with open(SPLIT_FILE, "wb") as f:
    pickle.dump(splits, f)

print(f"Done. Files saved in {OUTPUT_DIR}")
