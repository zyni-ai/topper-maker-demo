"""Shared data models and abstract interface for all HTR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image


class RegionType(str, Enum):
    TEXT = "text"
    MATH = "math"
    DIAGRAM = "diagram"
    TABLE = "table"
    UNKNOWN = "unknown"


@dataclass
class TextBlock:
    """One contiguous block of recognised text from the HTR engine."""
    text: str
    confidence: float            # 0.0 – 1.0
    region_type: RegionType = RegionType.TEXT
    # Normalised bounding box [x0, y0, x1, y1] in 0..1 coordinates
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class HTRResult:
    page_number: int
    blocks: list[TextBlock] = field(default_factory=list)
    engine: str = "unknown"
    raw_response: dict = field(default_factory=dict)
    truncated: bool = False  # True if the model response was cut off (finish_reason=length)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    @property
    def mean_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(b.confidence for b in self.blocks) / len(self.blocks)

    @property
    def min_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return min(b.confidence for b in self.blocks)


class BaseHTR(ABC):
    """Contract every HTR engine must fulfil."""

    @abstractmethod
    def extract(self, image: Image.Image, page_number: int = 1) -> HTRResult:
        """Run recognition on a PIL image; return HTRResult."""

