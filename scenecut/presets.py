"""Named parameter sets for the simple Detect node.

Presets are grouped by how hard the material is to cut, and every duration is in seconds so a
preset means the same thing at 24, 30 or 60 fps.

`film` and `music_video` reproduce the settings the original pipeline was measured with (12 and
5 frames at ~30 fps). The others start from reasoned values and are tuned on sample material.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Preset:
    label: str
    description: str
    # detection
    overlap: int = 20                    # frames shared by neighbouring 100-frame windows
    confidence: float = 0.0              # drop boundaries whose localisation peak is below this
    # refinement
    transitions: str = "merge_previous"  # merge_previous | merge_next | split | keep
    min_shot_seconds: float = 0.40       # shots shorter than this merge into a neighbour
    snap_radius: int = 2                 # frames either side a cut may move to its sharpest frame
    ultra_short_seconds: float = 0.05    # final clean-up of fragments
    # validation (DINOv3)
    validate: bool = True
    validate_window: int = 5             # frames either side of a cut that are compared
    flash_sensitivity: float = 15.0      # luma jump that counts as a flash; smaller = more flashes

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, Preset] = {
    "film": Preset(
        label="Film / TV",
        description="Measured pacing with occasional dissolves and fades. Shots under 0.4 s are "
                    "treated as detector noise.",
    ),
    "music_video": Preset(
        label="Music video / fast edit",
        description="Cuts on the beat, very short shots, strobes and flashes. Keeps shots down "
                    "to ~0.17 s and treats more light changes as flashes rather than cuts.",
        min_shot_seconds=0.17,
        flash_sensitivity=12.0,
    ),
    "anime": Preset(
        label="Anime / animation",
        description="Held drawings, impact flashes, smear frames and frequent fades to black. "
                    "Short shots kept down to 0.25 s; flashes detected aggressively.",
        min_shot_seconds=0.25,
        flash_sensitivity=10.0,
        validate_window=6,
    ),
    "vlog": Preset(
        label="Vlog / talking head",
        description="Jump cuts inside one setup and few real scene changes. Shots under 0.5 s "
                    "are merged.",
        min_shot_seconds=0.50,
    ),
    "sports": Preset(
        label="Sports / broadcast",
        description="Fast pans, graphic wipes into replays, logo stings. Transitions are kept as "
                    "their own segments so replays can be found.",
        transitions="keep",
        min_shot_seconds=0.30,
        flash_sensitivity=18.0,
    ),
}

DEFAULT_PRESET = "film"


def preset_names() -> list[str]:
    return list(PRESETS)


def get_preset(name: str, **overrides) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    p = PRESETS[name]
    return replace(p, **overrides) if overrides else p
