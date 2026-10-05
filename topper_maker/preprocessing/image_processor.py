"""OpenCV preprocessing: deskew → denoise → binarize → contrast enhance."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    deskew: bool = True
    denoise: bool = True
    binarize: bool = False      # Keep grayscale for HTR; binarize only if needed
    clahe: bool = True          # Contrast-limited adaptive histogram equalisation
    denoise_h: int = 10         # fastNlMeans filter strength
    clahe_clip: float = 2.0
    clahe_grid: tuple[int, int] = (8, 8)


class ImageProcessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def process(self, image: Image.Image) -> Image.Image:
        """Run the full preprocessing chain on a PIL Image, return PIL Image."""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        if self.config.deskew:
            gray = self._deskew(gray)
        if self.config.denoise:
            gray = self._denoise(gray)
        if self.config.clahe:
            gray = self._clahe(gray)
        if self.config.binarize:
            gray = self._binarize(gray)

        # Return as RGB PIL image so downstream code always gets the same type
        return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    # ------------------------------------------------------------------
    # Individual steps — each takes and returns a grayscale numpy array
    # ------------------------------------------------------------------

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Correct skew using the dominant line angle from Hough transform."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=gray.shape[1] // 4, maxLineGap=20,
        )
        if lines is None:
            return gray

        angles = [
            np.degrees(np.arctan2(y2 - y1, x2 - x1))
            for x1, y1, x2, y2 in lines[:, 0]
        ]
        # Keep only near-horizontal lines (within ±45°)
        horizontal = [a for a in angles if -45 < a < 45]
        if not horizontal:
            return gray

        median_angle = float(np.median(horizontal))
        if abs(median_angle) < 0.1:  # Already straight enough
            return gray

        h, w = gray.shape
        centre = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
        return cv2.warpAffine(
            gray, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, h=self.config.denoise_h)

    def _clahe(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip,
            tileGridSize=self.config.clahe_grid,
        )
        return clahe.apply(gray)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Sauvola-style adaptive threshold via OpenCV's adaptiveThreshold."""
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
