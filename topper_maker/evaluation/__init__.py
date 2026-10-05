"""Rubric-driven answer-sheet evaluation pipeline.

This package turns a scanned student answer sheet (PDF on S3) into per-question
marks and feedback, evaluated against a marking rubric in the style of the
Karnataka PU board. It is the evaluation layer built on top of the HTR
extraction components in :mod:`topper_maker.ingestion`,
:mod:`topper_maker.preprocessing`, and :mod:`topper_maker.htr`.

Public entry point: :class:`topper_maker.evaluation.pipeline.EvaluationPipeline`.
"""

from topper_maker.evaluation.pipeline import EvaluationPipeline
from topper_maker.evaluation.config import EvaluationConfig

__all__ = ["EvaluationPipeline", "EvaluationConfig"]
