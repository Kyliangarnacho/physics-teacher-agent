"""Stage 10 会话 Schema 与最近历史窗口单元测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.conversation import (
    ConversationRecord,
    ConversationState,
    StoredMessage,
    build_recent_history,
)


def stored_message(
    role: str,
    model_content: str,
    *,
    index: int = 0,
    display_content: str | None = None,
    image_metadata_json: object = None,
    created_at: str = "2026-08-06T08:00:00+00:00",
) -> StoredMessage:
    return StoredMessage(
        id=f"msg-{index}",
        conversation_id="conv-1",
        role=role,
        display_content=(
            display_content if display_content is not None else model_content
        ),
        model_content=model_content,
        image_metadata_json=image_metadata_json,
        created_at=created_at,
    )


def complete_turn(
    start: int,
    user_content: str,
    assistant_content: str,
) -> list[StoredMessage]:
    return [
        stored_message("user", user_content, index=start),
        stored_message("assistant", assistant_content, index=start + 1),
    ]


class BuildRecentHistoryTests(unittest.TestCase):
    def test_three_complete_turns_are_kept_in_order(self) -> None:
        messages = (
            complete_turn(0, "第一题", "第一答")
            + complete_turn(2, "第二题", "第二答")
            + complete_turn(4, "第三题", "第三答")
        )

        output = build_recent_history(messages)

        self.assertEqual(
            [(item["role"], item["content"]) for item in output],
            [
                ("user", "第一题"),
                ("assistant", "第一答"),
                ("user", "第二题"),
                ("assistant", "第二答"),
                ("user", "第三题"),
                ("assistant", "第三答"),
            ],
        )

    def test_more_turns_than_max_keeps_recent_only(self) -> None:
        messages = (
            complete_turn(0, "第一题", "第一答")
            + complete_turn(2, "第二题", "第二答")
            + complete_turn(4, "第三题", "第三答")
            + complete_turn(6, "第四题", "第四答")
        )

        output = build_recent_history(messages, max_turns=3)

        self.assertEqual(
            [item["content"] for item in output],
            ["第二题", "第二答", "第三题", "第三答", "第四题", "第四答"],
        )

    def test_char_budget_drops_oldest_complete_turn(self) -> None:
        messages = (
            complete_turn(0, "一", "甲")
            + complete_turn(2, "二二", "乙乙")
            + complete_turn(4, "三三三", "丙丙丙")
        )

        # 每轮字符数为 2、4、6，总计 12；预算 10 时只删除最旧一轮。
        output = build_recent_history(messages, max_chars=10)

        self.assertEqual(
            [item["content"] for item in output],
            ["二二", "乙乙", "三三三", "丙丙丙"],
        )

    def test_newest_turn_over_budget_returns_empty(self) -> None:
        messages = complete_turn(0, "长长长长长", "长长长")

        self.assertEqual(build_recent_history(messages, max_chars=5), [])

    def test_trailing_unanswered_user_is_ignored(self) -> None:
        messages = complete_turn(0, "第一题", "第一答") + [
            stored_message("user", "还没回答的问题", index=2)
        ]

        output = build_recent_history(messages)

        self.assertEqual(len(output), 2)
        self.assertEqual(output[0]["content"], "第一题")
        self.assertEqual(output[1]["content"], "第一答")

    def test_anomalous_role_order_is_not_re_paired(self) -> None:
        # assistant 开头 + 末尾未回答 user：不应拼出任何轮次。
        messages = [
            stored_message("assistant", "开头助手", index=0),
            stored_message("user", "孤立问题", index=1),
        ]
        self.assertEqual(build_recent_history(messages), [])

        # user 连续出现时只配对“后一个 user + assistant”，不跨角色拼接。
        messages = [
            stored_message("user", "被覆盖的问题", index=0),
            stored_message("user", "真正的问题", index=1),
            stored_message("assistant", "真正回答", index=2),
        ]
        output = build_recent_history(messages)
        self.assertEqual(
            [(item["role"], item["content"]) for item in output],
            [("user", "真正的问题"), ("assistant", "真正回答")],
        )

    def test_content_uses_model_content_not_display_content(self) -> None:
        messages = [
            stored_message(
                "user",
                "模型内容",
                index=0,
                display_content="显示内容",
            ),
            stored_message(
                "assistant",
                "模型回答",
                index=1,
                display_content="显示回答",
            ),
        ]

        output = build_recent_history(messages)

        self.assertEqual(output[0]["content"], "模型内容")
        self.assertEqual(output[1]["content"], "模型回答")
        self.assertNotIn("显示内容", output[0]["content"])

    def test_chinese_and_safe_image_context_stay_out_of_output(self) -> None:
        metadata = {
            "image_count": 1,
            "image_filenames": ["01.png"],
            "image_hash": ["abc123"],
            "image_context_used": True,
        }
        messages = [
            stored_message(
                "user",
                "看图回答串联电路问题",
                index=0,
                image_metadata_json=metadata,
            ),
            stored_message("assistant", "这是一张串联电路图", index=1),
        ]

        output = build_recent_history(messages)

        for item in output:
            self.assertEqual(set(item), {"role", "content"})
        self.assertEqual(output[0]["content"], "看图回答串联电路问题")
        self.assertEqual(output[1]["content"], "这是一张串联电路图")

    def test_zero_budget_returns_empty(self) -> None:
        messages = complete_turn(0, "问题", "回答")

        self.assertEqual(build_recent_history(messages, max_turns=0), [])
        self.assertEqual(build_recent_history(messages, max_chars=0), [])

    def test_negative_budget_raises_value_error(self) -> None:
        messages = complete_turn(0, "问题", "回答")

        with self.assertRaises(ValueError):
            build_recent_history(messages, max_turns=-1)
        with self.assertRaises(ValueError):
            build_recent_history(messages, max_chars=-1)

    def test_output_never_contains_db_json_trace_or_tool_fields(self) -> None:
        messages = complete_turn(0, "问题", "回答")

        output = build_recent_history(messages)

        for item in output:
            self.assertEqual(set(item), {"role", "content"})
        joined = "".join(item["content"] for item in output)
        for forbidden in ("trace", "sources", "tool_records", "image_metadata"):
            self.assertNotIn(forbidden, joined)


class ConversationSchemaTests(unittest.TestCase):
    def test_role_only_allows_user_and_assistant(self) -> None:
        stored_message("user", "内容")
        stored_message("assistant", "内容")

        with self.assertRaises(ValidationError):
            stored_message("system", "内容")
        with self.assertRaises(ValidationError):
            stored_message("tool", "内容")

    def test_hint_step_must_be_non_negative(self) -> None:
        ConversationState(
            conversation_id="conv-1",
            hint_step=0,
            updated_at="2026-08-06T08:00:00+00:00",
        )

        with self.assertRaises(ValidationError):
            ConversationState(
                conversation_id="conv-1",
                hint_step=-1,
                updated_at="2026-08-06T08:00:00+00:00",
            )

    def test_extra_fields_are_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            StoredMessage(
                id="msg-1",
                conversation_id="conv-1",
                role="user",
                display_content="内容",
                model_content="内容",
                created_at="2026-08-06T08:00:00+00:00",
                unexpected="x",
            )
        with self.assertRaises(ValidationError):
            ConversationRecord(
                id="conv-1",
                title="标题",
                created_at="2026-08-06T08:00:00+00:00",
                updated_at="2026-08-06T08:00:00+00:00",
                archived=0,
                extra=1,
            )
        with self.assertRaises(ValidationError):
            ConversationState(
                conversation_id="conv-1",
                updated_at="2026-08-06T08:00:00+00:00",
                extra=1,
            )

    def test_image_metadata_default_is_not_shared(self) -> None:
        first = stored_message("user", "内容", index=0)
        second = stored_message("user", "内容", index=1)

        first.image_metadata_json.append({"image_count": 1})

        self.assertEqual(second.image_metadata_json, [])
        self.assertIsNot(first.image_metadata_json, second.image_metadata_json)

    def test_image_metadata_dict_is_normalized_to_list(self) -> None:
        message = stored_message(
            "user",
            "内容",
            image_metadata_json={"image_count": 1, "image_filenames": ["01.png"]},
        )

        self.assertIsInstance(message.image_metadata_json, list)
        self.assertEqual(
            message.image_metadata_json,
            [{"image_count": 1, "image_filenames": ["01.png"]}],
        )

    def test_raw_image_bytes_base64_and_data_url_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            stored_message(
                "user",
                "内容",
                image_metadata_json=[{"bytes": b"raw"}],
            )
        with self.assertRaises(ValidationError):
            stored_message(
                "user",
                "内容",
                image_metadata_json=[
                    {"data": "data:image/png;base64,AAAA"}
                ],
            )
        with self.assertRaises(ValidationError):
            stored_message(
                "user",
                "内容",
                image_metadata_json=[{"base64": "AAAA"}],
            )


if __name__ == "__main__":
    unittest.main()
