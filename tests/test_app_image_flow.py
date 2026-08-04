"""No-API Streamlit tests for unified text and image submissions."""

from __future__ import annotations

from io import BytesIO
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from src.vision.schemas import ImageQuestionExtraction


def png_bytes(color: str = "white") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def image_item(name: str, data: bytes) -> dict[str, object]:
    return {"filename": name, "mime_type": "image/png", "bytes": data}


def service_result(status="complete", uncertain=None, name="image"):
    extraction = ImageQuestionExtraction(
        status=status,
        image_type="circuit",
        extracted_text=f"{name}：电压为 12 V，电阻为 6 Ω。",
        visual_elements=["电源", "电阻"],
        relationships=["电源与电阻构成闭合回路"],
        values_and_units=["12 V", "6 Ω"],
        formulas=[],
        student_work=[],
        uncertain_items=uncertain or [],
        suggested_user_question="求电流。",
        needs_ocr=False,
    )
    return {
        "prepared_image": {
            "image_hash": (name[0].lower() if name else "a") * 64,
            "mime_type": "image/png",
            "width": 24,
            "height": 16,
            "byte_size": 80,
            "resized": False,
        },
        "extraction": extraction,
        "ocr_used": False,
        "vision_run_id": f"vision-{name}",
        "vision_step_traces": [
            {
                "name": "vision_extract",
                "status": "success",
                "attempts": 1,
                "duration_ms": 1,
                "model_requests": 1,
                "metadata": {"data_url": "not-safe"},
            }
        ],
        "model_requests": 1,
    }


def agent_result() -> dict[str, object]:
    return {
        "answer": "通过电阻的电流为 2 A。",
        "analysis": {},
        "route": {},
        "sources": [],
        "analysis_fallback": False,
        "tool_records": [],
        "tool_model_requests": 0,
        "trace": None,
    }


def app_with_submission(text: str, pasted=None, attached=None) -> AppTest:
    app = AppTest.from_file("app.py")
    app.session_state["pending_chat_submission"] = {
        "text": text,
        "pasted_images": pasted or [],
        "attached_images": attached or [],
    }
    return app


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


class AppImageFlowTests(unittest.TestCase):
    def test_chat_input_enables_multiple_supported_files(self) -> None:
        app = AppTest.from_file("app.py")
        app.run()
        proto = app.chat_input[0].proto
        self.assertTrue(proto.accept_file)
        self.assertEqual(list(proto.file_type), [".jpg", ".jpeg", ".png", ".webp"])
        self.assertEqual(proto.max_upload_size_mb, 8)

    def test_plain_text_keeps_legacy_agent_call(self) -> None:
        calls = []

        def fake_agent(question, **kwargs):
            calls.append((question, kwargs))
            return agent_result()

        with patch("src.agent.run_teacher_agent", side_effect=fake_agent):
            app = AppTest.from_file("app.py")
            app.run()
            app.chat_input[0].set_value("为什么金属摸起来更凉？").run()

        self.assertEqual(calls[0][0], "为什么金属摸起来更凉？")
        self.assertEqual(
            calls[0][1],
            {"mode_override": "auto", "rag_policy": "auto"},
        )

    def test_text_and_multiple_images_run_once_and_preserve_order(self) -> None:
        vision_calls = []
        agent_calls = []

        def fake_vision(_bytes, filename, **_kwargs):
            vision_calls.append(filename)
            return service_result(name=filename)

        def fake_agent(question, **kwargs):
            agent_calls.append((question, kwargs))
            return agent_result()

        app = app_with_submission(
            "求两张图中的电流",
            pasted=[image_item("paste.png", png_bytes("white"))],
            attached=[image_item("attach.png", png_bytes("black"))],
        )
        with (
            patch("src.vision.batch.analyze_uploaded_image", side_effect=fake_vision),
            patch("src.agent.run_teacher_agent", side_effect=fake_agent),
        ):
            app.run()

        self.assertEqual(vision_calls, ["paste.png", "attach.png"])
        self.assertEqual(len(agent_calls), 1)
        context = agent_calls[0][1]["image_context"]
        self.assertLess(context.index("paste.png"), context.index("attach.png"))
        self.assertTrue(agent_calls[0][1]["image_context_available"])

    def test_duplicate_paste_and_attachment_are_sent_once(self) -> None:
        same_bytes = png_bytes("blue")
        vision_calls = []
        app = app_with_submission(
            "解答这张图",
            pasted=[image_item("pasted.png", same_bytes)],
            attached=[image_item("attached.png", same_bytes)],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                side_effect=lambda _bytes, filename, **_kwargs: (
                    vision_calls.append(filename) or service_result(name=filename)
                ),
            ),
            patch("src.agent.run_teacher_agent", return_value=agent_result()),
        ):
            app.run()

        self.assertEqual(vision_calls, ["pasted.png"])
        user_message = app.session_state["messages"][0]
        self.assertEqual(user_message["image_count"], 1)
        self.assertEqual(user_message["image_filenames"], ["pasted.png"])

    def test_image_only_uses_safe_default_question(self) -> None:
        calls = []
        app = app_with_submission(
            "",
            attached=[image_item("only.png", png_bytes())],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(name="only"),
            ),
            patch(
                "src.agent.run_teacher_agent",
                side_effect=lambda question, **kwargs: (
                    calls.append((question, kwargs)) or agent_result()
                ),
            ),
        ):
            app.run()
        self.assertEqual(
            calls[0][0],
            "请分析并解答这些图片中的物理问题。",
        )

    def test_more_than_three_images_stops_before_vision_and_agent(self) -> None:
        images = [
            image_item(f"{index}.png", bytes([index]))
            for index in range(1, 5)
        ]
        app = app_with_submission("测试", attached=images)
        with (
            patch("src.vision.batch.analyze_uploaded_image") as vision,
            patch("src.agent.run_teacher_agent") as agent,
        ):
            app.run()
        self.assertEqual(vision.call_count, 0)
        self.assertEqual(agent.call_count, 0)
        self.assertTrue(any("最多提交 3 张" in str(item.value) for item in app.warning))

    def test_confirmation_pauses_then_continues_without_reanalysis(self) -> None:
        app = app_with_submission(
            "请解答",
            attached=[image_item("check.png", png_bytes())],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(
                    status="needs_confirmation",
                    uncertain=["电表量程不清楚"],
                    name="check",
                ),
            ) as vision,
            patch("src.agent.run_teacher_agent", return_value=agent_result()) as agent,
        ):
            app.run()
            self.assertEqual(agent.call_count, 0)
            self.assertIsNotNone(app.session_state["pending_submission"])
            button(app, "确认并发送").click().run()

        self.assertEqual(vision.call_count, 1)
        self.assertEqual(agent.call_count, 1)
        self.assertNotIn("pending_submission", app.session_state)

    def test_unreadable_image_blocks_agent(self) -> None:
        app = app_with_submission(
            "请解答",
            attached=[image_item("bad.png", png_bytes())],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(status="unreadable", name="bad"),
            ),
            patch("src.agent.run_teacher_agent") as agent,
        ):
            app.run()
        self.assertEqual(agent.call_count, 0)
        self.assertTrue(any("无法识别" in str(item.value) for item in app.error))

    def test_cache_and_completed_batch_do_not_pollute_next_text_turn(self) -> None:
        agent_calls = []
        app = app_with_submission(
            "先看图",
            attached=[image_item("same.png", png_bytes())],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(name="same"),
            ) as vision,
            patch(
                "src.agent.run_teacher_agent",
                side_effect=lambda question, **kwargs: (
                    agent_calls.append((question, kwargs)) or agent_result()
                ),
            ),
        ):
            app.run()
            app.chat_input[0].set_value("下一轮纯文字").run()

        self.assertEqual(vision.call_count, 1)
        self.assertEqual(len(agent_calls), 2)
        self.assertIn("image_context", agent_calls[0][1])
        self.assertNotIn("image_context", agent_calls[1][1])

    def test_plain_rerun_after_image_send_does_not_repeat_vision(self) -> None:
        app = app_with_submission(
            "先看图",
            attached=[image_item("once.png", png_bytes())],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(name="once"),
            ) as vision,
            patch("src.agent.run_teacher_agent", return_value=agent_result()),
        ):
            app.run()
            app.run()
        self.assertEqual(vision.call_count, 1)

    def test_clear_conversation_removes_pending_batch_and_pasted_state(self) -> None:
        app = app_with_submission(
            "请核对",
            attached=[image_item("pending.png", png_bytes())],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(
                    status="needs_confirmation",
                    uncertain=["需核对"],
                    name="pending",
                ),
            ),
            patch("src.agent.run_teacher_agent"),
        ):
            app.run()
            app.session_state["paste_image_bridge"] = {
                "images": [
                    {
                        "filename": "later.png",
                        "mime_type": "image/png",
                        "bytes": [1, 2, 3],
                    }
                ]
            }
            button(app, "清空当前对话").click().run()

        self.assertEqual(app.session_state["messages"], [])
        self.assertNotIn("pending_submission", app.session_state)
        self.assertNotIn("pending_chat_submission", app.session_state)
        self.assertNotIn("paste_image_bridge", app.session_state)

    def test_history_and_cache_contain_no_bytes_context_or_data_url(self) -> None:
        app = app_with_submission(
            "请解答",
            attached=[image_item("safe.png", png_bytes())],
        )
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(name="safe"),
            ),
            patch("src.agent.run_teacher_agent", return_value=agent_result()),
        ):
            app.run()

        serialized = str(
            {
                "messages": app.session_state["messages"],
                "cache": app.session_state["vision_batch_cache"],
            }
        )
        self.assertNotIn("data_url", serialized)
        self.assertNotIn("base64", serialized.lower())
        self.assertNotIn("已确认图片上下文", serialized)
        self.assertNotIn("bytes", serialized)

    def test_native_chat_input_does_not_mount_duplicate_paste_preview(self) -> None:
        with patch(
            "src.agent.run_teacher_agent", return_value=agent_result()
        ) as agent:
            app = AppTest.from_file("app.py")
            app.run()
            app.chat_input[0].set_value("纯文字仍可用").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(agent.call_count, 1)
        self.assertFalse(
            any("粘贴暂不可用" in str(item.value) for item in app.caption)
        )


if __name__ == "__main__":
    unittest.main()
