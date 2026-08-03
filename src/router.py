"""初中物理教师 Agent 的教学模式、RAG 与工具路由规则。"""

from src.schemas import QuestionAnalysis, RouteDecision, TeachingMode


VALID_MODE_OVERRIDES = {"auto", *(mode.value for mode in TeachingMode)}
VALID_RAG_POLICIES = {"auto", "force", "off"}


def route_question(
    analysis: QuestionAnalysis,
    mode_override: str = "auto",
    rag_policy: str = "auto",
) -> RouteDecision:
    """根据结构化问题分析和用户策略生成路由决策。"""
    if mode_override not in VALID_MODE_OVERRIDES:
        raise ValueError(
            "mode_override 必须是 auto、solve、explain、hint 或 diagnose。"
        )
    if rag_policy not in VALID_RAG_POLICIES:
        raise ValueError("rag_policy 必须是 auto、force 或 off。")

    teaching_mode = (
        analysis.teaching_mode
        if mode_override == "auto"
        else TeachingMode(mode_override)
    )

    if analysis.image_required:
        return RouteDecision(
            teaching_mode=teaching_mode,
            use_rag=False,
            use_tools=False,
            should_answer=False,
            user_message="请补充题图，或完整描述图中的物体、连接关系和已知信息。",
        )

    if rag_policy == "auto":
        use_rag = analysis.needs_rag
    else:
        use_rag = rag_policy == "force"
    use_tools = analysis.calculation_required and not analysis.missing_conditions

    return RouteDecision(
        teaching_mode=teaching_mode,
        use_rag=use_rag,
        use_tools=use_tools,
        should_answer=True,
    )
