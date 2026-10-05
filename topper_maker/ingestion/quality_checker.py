"""Image quality validation: blur, brightness, contrast, blank-page, orientation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import cv2
import numpy as np
from PIL import Image

from topper_maker.ingestion.pdf_extractor import PageImage


class Orientation(str, Enum):
    UPRIGHT = "upright"
    ROTATED_90 = "rotated_90"
    ROTATED_180 = "rotated_180"
    ROTATED_270 = "rotated_270"
    UNKNOWN = "unknown"


@dataclass
class QualityReport:
    page_number: int
    dpi: int                  # Nominal DPI from config, not measured from scan
    blur_score: float         # Laplacian variance — higher = sharper
    brightness: float         # Mean pixel intensity 0-255
    contrast: float           # Std-dev of pixel intensities
    is_blank: bool            # True when the page has no meaningful ink
    orientation: Orientation  # Detected page orientation
    passed: bool
    warnings: list[str]

    # Thresholds mirroring QualityChecker defaults — ClassVar so they are not
    # dataclass fields (excluded from __init__, __repr__, and serialisation).
    MIN_BLUR_SCORE: ClassVar[float] = 50.0
    MIN_DPI: ClassVar[int] = 200
    MIN_BRIGHTNESS: ClassVar[float] = 50.0
    MAX_BRIGHTNESS: ClassVar[float] = 253.0
    MIN_CONTRAST: ClassVar[float] = 20.0
    BLANK_PAGE_CONTRAST_THRESHOLD: ClassVar[float] = 8.0


class QualityChecker:
    def __init__(
        self,
        min_blur_score: float = 50.0,
        min_dpi: int = 200,
        min_brightness: float = 50.0,
        max_brightness: float = 253.0,
        min_contrast: float = 20.0,
        blank_contrast_threshold: float = 8.0,
    ) -> None:
        self.min_blur_score = min_blur_score
        self.min_dpi = min_dpi
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.blank_contrast_threshold = blank_contrast_threshold

    def check(self, page: PageImage) -> QualityReport:
        gray = self._to_gray(page.image)
        blur_score = self._blur_score(gray)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        is_blank = self._is_blank(contrast)
        orientation = self._detect_orientation(page.image)

        warnings: list[str] = []

        # NOTE: page.dpi is the target DPI from config, not the actual scan resolution.
        # A low value here means the pipeline was configured with a low DPI, not that
        # the scan itself is low resolution.
        if page.dpi < self.min_dpi:
            warnings.append(
                f"Target DPI {page.dpi} is below minimum {self.min_dpi}. "
                "Re-render the PDF or rescan at 300 DPI."
            )

        if is_blank:
            warnings.append(
                "Page appears blank (very low ink contrast). "
                "Check for a misaligned scan or an intentionally empty page."
            )

        if not is_blank:
            if blur_score < self.min_blur_score:
                warnings.append(
                    f"Blur score {blur_score:.1f} below threshold {self.min_blur_score}; "
                    "image may be out of focus."
                )
            if brightness < self.min_brightness:
                warnings.append(f"Image too dark (brightness={brightness:.1f}).")
            if brightness > self.max_brightness:
                warnings.append(f"Image too bright/washed out (brightness={brightness:.1f}).")
            if contrast < self.min_contrast:
                warnings.append(
                    f"Low contrast ({contrast:.1f}); ink may be faded or paper too light."
                )

        if orientation != Orientation.UPRIGHT and orientation != Orientation.UNKNOWN:
            warnings.append(
                f"Page may be rotated ({orientation.value}). "
                "Preprocessing deskew cannot correct 90°/180° rotation — rescan or rotate manually."
            )

        return QualityReport(
            page_number=page.page_number,
            dpi=page.dpi,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            is_blank=is_blank,
            orientation=orientation,
            passed=len(warnings) == 0,
            warnings=warnings,
        )

    def check_all(self, pages: list[PageImage]) -> list[QualityReport]:
        return [self.check(p) for p in pages]

    def _is_blank(self, contrast: float) -> bool:
        return contrast < self.blank_contrast_threshold

    @staticmethod
    def _to_gray(image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        """Laplacian variance — a well-known sharpness proxy."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Ratio of the two axes' ink-profile coefficients of variation above which we
    # call the dominant text direction. Text that runs left-to-right makes the ink
    # count swing sharply from row to row (line, gap, line, …) while staying smooth
    # column to column; rotated 90°/270° text inverts that. 1.25 keeps genuinely
    # ambiguous pages (diagrams, sparse ink) in UNKNOWN rather than guessing.
    _ORIENTATION_RATIO = 1.25

    @staticmethod
    def _detect_orientation(image: Image.Image) -> Orientation:
        """Estimate page orientation from the ink projection profiles (issue #38).

        Replaces the old Hough-line heuristic, which returned ``UNKNOWN`` on almost
        every real (handwritten) page because handwriting yields few long straight
        lines. We instead compare how much the ink density varies row-to-row versus
        column-to-column:

        - horizontal-dominant variation → text lines run across the page → ``UPRIGHT``
        - vertical-dominant variation → text lines run down the page → ``ROTATED_90``
        - comparable in both axes, or no ink → ``UNKNOWN``

        Limitation: this distinguishes the text *axis* only. It cannot tell ``UPRIGHT``
        from 180° (both horizontal) or 90° from 270° (both vertical) without reading
        the text, so an upside-down page reports ``UPRIGHT``. ``ROTATED_90`` should be
        read as "rotated a quarter turn (90° or 270°)".
        """
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Ink = dark pixels. Otsu adapts to scan brightness; INV so ink becomes 1.
        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if binary.sum() < gray.size * 0.001:  # essentially no ink — let blank check own it
            return Orientation.UNKNOWN

        row_profile = binary.sum(axis=1).astype(np.float64)  # ink per row
        col_profile = binary.sum(axis=0).astype(np.float64)  # ink per column
        cov_rows = QualityChecker._coeff_of_variation(row_profile)
        cov_cols = QualityChecker._coeff_of_variation(col_profile)

        if cov_rows == 0 and cov_cols == 0:
            return Orientation.UNKNOWN
        # A near-zero CoV on one axis means the ink is perfectly even along it, i.e.
        # the other axis fully dominates — the clearest possible signal, not an
        # ambiguous one. Handle it before the ratio (which would divide by ~zero).
        if cov_cols == 0:
            return Orientation.UPRIGHT
        if cov_rows == 0:
            return Orientation.ROTATED_90
        if cov_rows / cov_cols >= QualityChecker._ORIENTATION_RATIO:
            return Orientation.UPRIGHT
        if cov_cols / cov_rows >= QualityChecker._ORIENTATION_RATIO:
            return Orientation.ROTATED_90
        return Orientation.UNKNOWN

    @staticmethod
    def _coeff_of_variation(profile: np.ndarray) -> float:
        """std/mean — scale-free spread, so page size and ink density cancel out."""
        mean = float(profile.mean())
        return float(profile.std() / mean) if mean > 0 else 0.0
