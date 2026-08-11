"""初中物理教师 Agent 的结构化分析与路由数据模型。"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    model_validator,
)


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ShortReasonText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]


class TeachingMode(str, Enum):
    """教师回答问题时采用的教学方式。"""

    SOLVE = "solve"
    EXPLAIN = "explain"
    HINT = "hint"
    DIAGNOSE = "diagnose"


class ContextRelation(str, Enum):
    """当前问题与上一活动题目的语义关系。"""

    FOLLOW_UP = "follow_up"
    NEW_PROBLEM = "new_problem"
    UNCERTAIN = "uncertain"


class RunStatus(str, Enum):
    """一次 Agent 运行的最终状态。"""

    COMPLETED = "completed"
    COMPLETED_WITH_FALLBACK = "completed_with_fallback"
    BLOCKED = "blocked"
    FAILED = "failed"


class StepStatus(str, Enum):
    """一次运行步骤的执行状态。"""

    SUCCESS = "success"
    RETRY_SUCCESS = "retry_success"
    SKIPPED = "skipped"
    ERROR = "error"


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
    calculation_required: StrictBool = False
    context_relation: ContextRelation = ContextRelation.UNCERTAIN
    needs_previous_image_context: StrictBool = False


class RouteDecision(BaseModel):
    """根据问题分析得到的最小路由决策。"""

    model_config = ConfigDict(extra="forbid")

    teaching_mode: TeachingMode
    use_rag: bool
    should_answer: bool
    user_message: str | None = None
    use_tools: StrictBool = False


class StepTrace(BaseModel):
    """一个可序列化的 Agent 执行步骤记录。"""

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyText
    status: StepStatus
    attempts: NonNegativeInt
    duration_ms: NonNegativeFloat
    model_requests: NonNegativeInt = 0
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_constraints(self) -> "StepTrace":
        if self.status is StepStatus.SKIPPED:
            if self.attempts != 0 or self.model_requests != 0:
                raise ValueError("skipped 步骤的 attempts 和 model_requests 必须为 0。")
        elif self.status in {StepStatus.SUCCESS, StepStatus.ERROR}:
            if self.attempts < 1:
                raise ValueError("success 和 error 步骤的 attempts 必须至少为 1。")
        elif self.status is StepStatus.RETRY_SUCCESS and self.attempts != 2:
            raise ValueError("retry_success 步骤的 attempts 必须为 2。")

        has_error_type = self.error_type is not None
        has_error_message = self.error_message is not None
        if self.status is StepStatus.ERROR:
            if not has_error_type or not has_error_message:
                raise ValueError("error 步骤必须同时提供 error_type 和 error_message。")
        elif has_error_type or has_error_message:
            raise ValueError("非 error 步骤不能携带最终错误字段。")
        return self


class ContextTraceMetadata(BaseModel):
    """Safe, content-free ContextBundle statistics for one Agent run."""

    model_config = ConfigDict(extra="forbid", strict=True)

    summary_revision: PositiveInt | None
    summary_present: StrictBool
    bridge_turn_count: NonNegativeInt
    recent_turn_count: NonNegativeInt
    retrieved_turn_count: NonNegativeInt
    history_retrieval_used: StrictBool
    context_estimated_chars: NonNegativeInt
    budget_limit: PositiveInt
    budget_exceeded: StrictBool
    trimmed_components: list[
        Literal[
            "retrieved_history",
            "learning_memory_context",
            "rolling_summary",
            "unsafe_history",
        ]
    ]
    analyzer_context_chars: NonNegativeInt
    tool_context_chars: NonNegativeInt | None
    final_context_chars: NonNegativeInt | None

    @model_validator(mode="after")
    def validate_trimmed_components(self) -> "ContextTraceMetadata":
        if len(self.trimmed_components) != len(set(self.trimmed_components)):
            raise ValueError("trimmed_components must not contain duplicates.")
        return self


class AgentRunTrace(BaseModel):
    """一次 Agent 运行的结构化汇总记录。"""

    model_config = ConfigDict(extra="forbid")

    run_id: NonEmptyText
    case_id: str | None = None
    started_at: NonEmptyText
    finished_at: NonEmptyText
    total_duration_ms: NonNegativeFloat
    status: RunStatus
    total_model_requests: NonNegativeInt
    rag_searches: NonNegativeInt
    tool_executions: NonNegativeInt
    analysis_fallback: StrictBool
    teaching_mode: TeachingMode | None
    use_rag: StrictBool
    use_tools: StrictBool
    context_metadata: ContextTraceMetadata | None = None
    steps: list[StepTrace] = Field(default_factory=list)
