"""VASI (Vitiligo Area Scoring Index) computation.

VASI = sum over body regions of  hand_units(region) x depigmentation_fraction(region)

- hand_units: approximate size of each body region in "hand units" (1 HU ≈ 1% BSA),
  following the Hamzavi et al. 2004 convention.
- depigmentation_fraction: fraction of the region that is depigmented (0–1), taken
  from the segmentation mask, weighted by depigmentation grade.

The grade weighting maps mask confidence to the clinical categories
(100 % / 90 % / 75 % / 50 % / 25 % / 10 %). With a binary mask, grade = 1.0.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Hand units per region (approximate, symmetric sides share the same value).
HAND_UNITS: dict[str, float] = {
    "face": 3.0, "neck": 1.0,
    "hand_dorsal_l": 0.5, "hand_dorsal_r": 0.5, "hand_palmar_l": 0.5, "hand_palmar_r": 0.5,
    "forearm_l": 3.0, "forearm_r": 3.0, "upper_arm_l": 4.0, "upper_arm_r": 4.0,
    "trunk_front": 13.0, "trunk_back": 13.0,
    "thigh_l": 9.0, "thigh_r": 9.0, "shin_l": 6.0, "shin_r": 6.0,
    "foot_l": 1.5, "foot_r": 1.5,
    "lesion_site": 0.0,  # ad-hoc close-ups are not scored; use them for change detection
}

@dataclass
class RegionScore:
    region: str
    depig_fraction: float      # 0–1, fraction of visible skin that is depigmented
    hand_units: float
    vasi_contribution: float   # hand_units * depig_fraction

def region_score(region: str, lesion_mask: np.ndarray, skin_mask: np.ndarray) -> RegionScore:
    """Score one region from a lesion mask and a skin mask (both bool HxW)."""
    if region not in HAND_UNITS:
        raise ValueError(f"unknown region {region!r}")
    skin_px = int(skin_mask.sum())
    lesion_px = int((lesion_mask & skin_mask).sum())
    frac = lesion_px / skin_px if skin_px else 0.0
    hu = HAND_UNITS[region]
    return RegionScore(region, frac, hu, hu * frac)

def total_vasi(scores: list[RegionScore]) -> float:
    return float(sum(s.vasi_contribution for s in scores))
