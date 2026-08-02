from __future__ import annotations

import json
from collections.abc import Callable

from openai import OpenAI

from src.config import load_qwen_config
from src.prompts import QUESTION_ANALYZER_SYSTEM_PROMPT
from src.schemas import QuestionAnalysis, TeachingMode


AnalyzeFunc = Callable[[str], str]


def _safe_default_analysis() -> QuestionAnalysis:
    return QuestionAnalysis(
        teaching_mode=TeachingMode.SOLVE,
        physics_topic="综合",
        question_type="未知",
        needs_rag=False,
        missing_conditions=False,
        image_required=False,
        student_work_provided=False,
        short_reason="问题分析失败，按普通完整解题处理。",
        calculation_required=False,
    )


def _call_analyzer_model(question: str) -> str:
    api_key, base_url, model = load_qwen_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUESTION_ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("问题分析模型返回了空内容。")
    return content


def analyze_question(
    question: str,
    analyze_func: AnalyzeFunc | None = None,
) -> tuple[QuestionAnalysis, bool]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空，请提供一道初中物理问题。")

    normalized_question = question.strip()
    try:
        raw_result = (
            analyze_func(normalized_question)
            if analyze_func is not None
            else _call_analyzer_model(normalized_question)
        )
        payload = json.loads(raw_result)
        analysis = QuestionAnalysis.model_validate(payload)
    except Exception:
        return _safe_default_analysis(), True

    return analysis, False
