"""Stage 06 的结构化数据模型。"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ShortReasonText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]


class TeachingMode(str, Enum):
    """教师回答问题时采用的教学方式。"""

    SOLVE = "solve"
    EXPLAIN = "explain"
    HINT = "hint"
    DIAGNOSE = "diagnose"


class QuestionAnalysis(BaseModel):
    """对用户物理问题的结构化分析结果。"""

    model_config = ConfigDict(extra="forbid")

    teaching_mode: TeachingMode
    physics_topic: NonEmptyText
    question_type: NonEmptyText
    needs_rag: bool
    missing_conditions: bool
    image_required: bool
    student_work_provided: bool
    short_reason: ShortReasonText


class RouteDecision(BaseModel):
    """根据问题分析得到的最小路由决策。"""

    model_config = ConfigDict(extra="forbid")

    teaching_mode: TeachingMode
    use_rag: bool
    should_answer: bool
    user_message: str | None = None
