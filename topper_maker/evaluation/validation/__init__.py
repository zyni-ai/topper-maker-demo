"""HTR validation tooling (issue #10): measure transcription quality on real scans."""

from topper_maker.evaluation.validation.htr_metrics import (
    detect_question_numbers,
    evaluate_criteria,
    mean_confidence,
    normalize_text,
    question_recall,
    summarize_failure_modes,
    text_error_rates,
)

__all__ = [
    "detect_question_numbers",
    "question_recall",
    "normalize_text",
    "text_error_rates",
    "mean_confidence",
    "summarize_failure_modes",
    "evaluate_criteria",
]
