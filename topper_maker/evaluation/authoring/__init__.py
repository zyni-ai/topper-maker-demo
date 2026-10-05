"""Paper/rubric authoring: compile a question paper + answer key into evaluation inputs.

The :class:`PaperCompiler` reads a question-paper PDF and an answer-key PDF and emits a
reviewable :class:`CompiledPaper` whose ``questions`` + ``optional_sections`` feed
straight into an ``EvaluationRequest``. This is the input-side counterpart to the
:class:`~topper_maker.evaluation.pipeline.EvaluationPipeline`.
"""

from topper_maker.evaluation.authoring.paper_compiler import PaperCompiler
from topper_maker.evaluation.authoring.reconcile import reconcile
from topper_maker.evaluation.authoring.schemas import (
    CompiledPaper,
    KeyExtraction,
    PaperExtraction,
)

__all__ = [
    "PaperCompiler",
    "reconcile",
    "CompiledPaper",
    "PaperExtraction",
    "KeyExtraction",
]
