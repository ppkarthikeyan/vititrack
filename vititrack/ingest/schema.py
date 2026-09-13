"""Session metadata schema (Imaging Protocol v1)."""
from __future__ import annotations
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

Region = Literal[
    "face", "neck", "hand_dorsal_l", "hand_dorsal_r", "hand_palmar_l", "hand_palmar_r",
    "forearm_l", "forearm_r", "upper_arm_l", "upper_arm_r", "trunk_front", "trunk_back",
    "thigh_l", "thigh_r", "shin_l", "shin_r", "foot_l", "foot_r", "lesion_site",
]

class TreatmentEntry(BaseModel):
    name: str                      # e.g. "ruxolitinib 1.5% cream", "NB-UVB"
    started: date | None = None
    stopped: date | None = None
    notes: str = ""

class SessionMetadata(BaseModel):
    patient_id: str
    session_date: date
    phone_model: str
    lighting: Literal["daylight", "room", "mixed"] = "room"
    regions: list[Region] = Field(min_length=1)
    uv_captured: bool = True
    colour_card_present: bool = True
    treatments: list[TreatmentEntry] = []
    new_spots_reported: bool = False
    itch: bool = False
    trauma_sites: list[str] = []
    notes: str = ""
