"""Minimal Components v2 bridge for pasting images near st.chat_input."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import streamlit as st


MAX_PASTED_IMAGES = 3
MAX_PASTED_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_PASTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

_PASTE_BRIDGE_HTML = """
<div id="paste-strip" role="status" aria-live="polite"></div>
"""

_PASTE_BRIDGE_CSS = """
#paste-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  min-height: 0;
  color: var(--st-text-color);
  font: 0.82rem var(--st-font);
}
.paste-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  max-width: 12rem;
  padding: 0.28rem 0.45rem;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 14%, transparent);
  border-radius: 0.7rem;
  background: var(--st-secondary-background-color);
}
.paste-item img { width: 2rem; height: 2rem; border-radius: 0.4rem; object-fit: cover; }
.paste-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.paste-remove { border: 0; background: transparent; color: inherit; cursor: pointer; }
"""

_PASTE_BRIDGE_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component
  const strip = parentElement.querySelector("#paste-strip")
  if (!strip) return

  let images = Array.isArray(data?.images) ? data.images.slice(0, data.maxImages) : []
  const objectUrls = []

  const hashBytes = async bytes => {
    const digest = await crypto.subtle.digest("SHA-256", new Uint8Array(bytes))
    return Array.from(new Uint8Array(digest))
      .map(value => value.toString(16).padStart(2, "0"))
      .join("")
  }

  const render = () => {
    strip.replaceChildren()
    objectUrls.splice(0).forEach(url => URL.revokeObjectURL(url))
    images.forEach((item, index) => {
      const row = document.createElement("div")
      row.className = "paste-item"
      const preview = document.createElement("img")
      const bytes = new Uint8Array(item.bytes || [])
      const url = URL.createObjectURL(new Blob([bytes], { type: item.mime_type }))
      objectUrls.push(url)
      preview.src = url
      preview.alt = "待发送图片"
      const name = document.createElement("span")
      name.className = "paste-name"
      name.textContent = String(item.filename || `pasted-${index + 1}.png`)
      const remove = document.createElement("button")
      remove.className = "paste-remove"
      remove.type = "button"
      remove.setAttribute("aria-label", "删除粘贴图片")
      remove.textContent = "×"
      remove.onclick = () => {
        images = images.filter((_, itemIndex) => itemIndex !== index)
        setStateValue("images", images)
        render()
      }
      row.append(preview, name, remove)
      strip.append(row)
    })
  }

  const onPaste = async event => {
    const textarea = document.querySelector('[data-testid="stChatInput"] textarea')
    if (!textarea || document.activeElement !== textarea) return
    const files = Array.from(event.clipboardData?.files || [])
      .filter(file => file.type.startsWith("image/"))
    if (!files.length) return
    event.preventDefault()
    event.stopImmediatePropagation()
    const hashes = new Set(await Promise.all(
      images.map(item => hashBytes(item.bytes || []))
    ))
    for (const file of files) {
      if (images.length >= data.maxImages) break
      if (!data.allowedTypes.includes(file.type) || file.size > data.maxBytes) continue
      const bytes = Array.from(new Uint8Array(await file.arrayBuffer()))
      const hash = await hashBytes(bytes)
      if (hashes.has(hash)) continue
      hashes.add(hash)
      images.push({
        filename: file.name || `pasted-${Date.now()}.png`,
        mime_type: file.type,
        bytes,
      })
    }
    setStateValue("images", images)
    render()
  }

  document.addEventListener("paste", onPaste, true)
  render()
  return () => {
    document.removeEventListener("paste", onPaste, true)
    objectUrls.forEach(url => URL.revokeObjectURL(url))
  }
}
"""


class PasteBridgeUnavailable(RuntimeError):
    """Raised when the optional inline component cannot be mounted."""


try:
    _PASTE_COMPONENT = st.components.v2.component(
        "physics_teacher_paste_images",
        html=_PASTE_BRIDGE_HTML,
        css=_PASTE_BRIDGE_CSS,
        js=_PASTE_BRIDGE_JS,
    )
except Exception:
    _PASTE_COMPONENT = None


def decode_pasted_images(value: object) -> list[dict[str, Any]]:
    """Validate component state and immediately convert byte arrays to bytes."""
    if value is None:
        return []
    if isinstance(value, dict):
        raw_images = value.get("images", [])
    else:
        raw_images = getattr(value, "images", [])
    if not isinstance(raw_images, list):
        return []

    decoded: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for item in raw_images:
        if len(decoded) >= MAX_PASTED_IMAGES:
            break
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        mime_type = item.get("mime_type")
        raw_bytes = item.get("bytes")
        if (
            not isinstance(filename, str)
            or not filename.strip()
            or mime_type not in SUPPORTED_PASTED_MIME_TYPES
            or not isinstance(raw_bytes, list)
            or not raw_bytes
            or len(raw_bytes) > MAX_PASTED_IMAGE_BYTES
        ):
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in raw_bytes
        ):
            continue
        image_bytes = bytes(raw_bytes)
        image_hash = sha256(image_bytes).hexdigest()
        if image_hash in seen_hashes:
            continue
        seen_hashes.add(image_hash)
        decoded.append(
            {
                "filename": filename.strip(),
                "mime_type": mime_type,
                "bytes": image_bytes,
            }
        )
    return decoded


def remove_pasted_image(
    images: list[dict[str, Any]],
    index: int,
) -> list[dict[str, Any]]:
    """Return an independent list without one pasted image."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("index 必须是整数。")
    if not 0 <= index < len(images):
        raise ValueError("index 超出图片范围。")
    return [dict(item) for item_index, item in enumerate(images) if item_index != index]


def paste_image_bridge(*, key: str, reset_token: int = 0) -> list[dict[str, Any]]:
    """Mount the inline paste bridge and return validated pending images."""
    if _PASTE_COMPONENT is None:
        raise PasteBridgeUnavailable("图片粘贴桥接暂不可用。")
    current_images = [
        {
            "filename": item["filename"],
            "mime_type": item["mime_type"],
            "bytes": list(item["bytes"]),
        }
        for item in decode_pasted_images(st.session_state.get(key, {}))
    ]
    result = _PASTE_COMPONENT(
        key=key,
        data={
            "images": current_images,
            "maxImages": MAX_PASTED_IMAGES,
            "maxBytes": MAX_PASTED_IMAGE_BYTES,
            "allowedTypes": sorted(SUPPORTED_PASTED_MIME_TYPES),
            "resetToken": reset_token,
        },
        on_images_change=lambda: None,
        width="stretch",
        height="content",
    )
    return decode_pasted_images(result)
