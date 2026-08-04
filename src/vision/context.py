"""Pure helpers for turning visual extraction into user-editable context."""

from __future__ import annotations

from collections.abc import Iterable

from src.vision.schemas import ImageQuestionExtraction


def _list_section(title: str, values: Iterable[str]) -> str | None:
    items = [value.strip() for value in values if value.strip()]
    if not items:
        return None
    return f"【{title}】\n" + "\n".join(f"- {item}" for item in items)


def _text_section(title: str, value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    return f"【{title}】\n{normalized}"


def build_image_context_draft(
    extraction: ImageQuestionExtraction,
) -> str:
    """Build an editable, text-only context draft from one extraction."""
    if not isinstance(extraction, ImageQuestionExtraction):
        raise ValueError("extraction 必须是 ImageQuestionExtraction。")

    sections = [
        _text_section("题目文字", extraction.extracted_text),
        _list_section("图中元素", extraction.visual_elements),
        _list_section("图形关系", extraction.relationships),
        _list_section("数值与单位", extraction.values_and_units),
        _list_section("公式", extraction.formulas),
        _list_section("学生过程", extraction.student_work),
        _list_section("不确定内容", extraction.uncertain_items),
        _text_section("建议问题", extraction.suggested_user_question),
    ]
    return "\n\n".join(section for section in sections if section)
