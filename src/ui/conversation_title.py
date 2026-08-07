"""可双击编辑的会话标题小组件。"""

from __future__ import annotations

import streamlit as st


_TITLE_HTML = """
<div id="conversation-title"></div>
"""

_TITLE_CSS = """
#conversation-title { min-height: 2rem; display: flex; align-items: center; }
.conversation-title-label {
  width: 100%; padding: 0.35rem 0.45rem; border-radius: 0.55rem;
  cursor: text; color: var(--st-text-color); line-height: 1.35;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.conversation-title-label:hover { background: color-mix(in srgb, var(--st-text-color) 6%, transparent); }
.conversation-title-label:focus-visible { outline: 2px solid var(--st-primary-color); outline-offset: 1px; }
.conversation-current { margin-left: 0.3rem; color: var(--st-primary-color); font-size: 0.78em; }
.conversation-title-input {
  box-sizing: border-box; width: 100%; padding: 0.35rem 0.45rem;
  border: 1px solid var(--st-primary-color); border-radius: 0.55rem;
  background: var(--st-secondary-background-color); color: var(--st-text-color);
  font: inherit; outline: none;
}
"""

_TITLE_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const root = parentElement.querySelector("#conversation-title")
  if (!root) return

  const title = String(data?.title ?? "")
  const current = Boolean(data?.current)

  const showLabel = () => {
    root.replaceChildren()
    const label = document.createElement("div")
    label.className = "conversation-title-label"
    label.tabIndex = 0
    label.title = "双击修改会话名称"
    label.textContent = title
    if (current) {
      const marker = document.createElement("span")
      marker.className = "conversation-current"
      marker.textContent = "当前"
      label.append(marker)
    }
    const beginEdit = () => {
      root.replaceChildren()
      const input = document.createElement("input")
      input.className = "conversation-title-input"
      input.type = "text"
      input.value = title
      input.setAttribute("aria-label", "会话名称")
      input.addEventListener("keydown", event => {
        if (event.key === "Enter") {
          const next = input.value.trim()
          if (next && next !== title) setTriggerValue("submitted", next)
          else showLabel()
        } else if (event.key === "Escape") {
          showLabel()
        }
      })
      input.addEventListener("blur", showLabel)
      root.append(input)
      input.focus()
      input.select()
    }
    label.addEventListener("dblclick", event => {
      event.preventDefault()
      beginEdit()
    })
    label.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault()
        beginEdit()
      }
    })
    root.append(label)
  }

  showLabel()
}
"""


def _title_component():
    """在当前 Streamlit 脚本运行上下文注册组件。

    AppTest 会创建独立的组件注册表；不能复用模块首次导入时的 renderer。
    """
    return st.components.v2.component(
        "physics_teacher_conversation_title",
        html=_TITLE_HTML,
        css=_TITLE_CSS,
        js=_TITLE_JS,
    )


def editable_conversation_title(
    title: str,
    *,
    current: bool,
    key: str,
) -> str | None:
    """显示标题；双击后按 Enter 时返回新的非空标题。"""
    if not isinstance(title, str):
        raise ValueError("title 必须是字符串。")
    result = _title_component()(
        key=key,
        data={"title": title, "current": current},
        on_submitted_change=lambda: None,
        width="stretch",
        height="content",
    )
    submitted = getattr(result, "submitted", None)
    if not isinstance(submitted, str):
        return None
    normalized = submitted.strip()
    return normalized if normalized and normalized != title.strip() else None
