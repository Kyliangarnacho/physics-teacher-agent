"""Unified ContextBundle assembly and snapshot tests."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import src.context.manager as context_manager_module
from src.context.manager import (
    build_context_bundle,
    format_context_bundle_for_model,
)
from src.memory.schemas import normalize_memory_content
from src.storage import (
    create_conversation,
    initialize_database,
    insert_message,
    insert_or_merge_memory,
    upsert_conversation_state,
    upsert_conversation_summary,
)


CONTENTS = [
    ("并联电路支路电阻变化", "旧回答讨论支路电流"),
    ("凸透镜焦距与物距", "旧回答讨论成像位置"),
    ("浮力与排液体积", "旧回答使用阿基米德原理"),
    ("弹簧振子共振", "旧回答讨论固有频率"),
    ("桥接轮次斜面效率", "尚未进入摘要的旧回答"),
    ("最近轮次声速", "最近回答一"),
    ("最近轮次光的折射", "最近回答二"),
    ("最近轮次电磁铁", "最近回答三"),
]


class ContextManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "context-manager.db")
        initialize_database(self.db_path)

    def _store_conversation(
        self,
        title: str,
        contents: list[tuple[str, str]] | None = None,
    ) -> str:
        conversation_id = create_conversation(title, path=self.db_path)["id"]
        for index, (user_content, assistant_content) in enumerate(
            contents or CONTENTS,
            1,
        ):
            insert_message(
                {
                    "id": f"{conversation_id}-u{index}",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "display_content": user_content,
                    "model_content": user_content,
                },
                path=self.db_path,
            )
            insert_message(
                {
                    "id": f"{conversation_id}-a{index}",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "display_content": assistant_content,
                    "model_content": assistant_content,
                },
                path=self.db_path,
            )
        return conversation_id

    def _summarize(self, conversation_id: str, boundary: int, text: str = "旧摘要"):
        return upsert_conversation_summary(
            conversation_id,
            text,
            covered_until_message_id=f"{conversation_id}-a{boundary}",
            covered_turn_count=boundary,
            model_name="qwen-test",
            path=self.db_path,
        )

    def _add_state_and_memory(self, conversation_id: str) -> None:
        upsert_conversation_state(
            {
                "conversation_id": conversation_id,
                "active_problem_text": "凸透镜焦距与物距关系",
                "active_image_context": "图中物体位于二倍焦距外",
                "teaching_mode": "hint",
                "hint_step": 2,
            },
            path=self.db_path,
        )
        content = "学生曾混淆凸透镜的焦距和物距。"
        insert_or_merge_memory(
            {
                "memory_type": "misconception",
                "topic": "凸透镜",
                "content": content,
                "normalized_content": normalize_memory_content(content),
                "confidence": 0.9,
                "confirmed": 1,
                "active": 1,
                "source_conversation_id": conversation_id,
            },
            path=self.db_path,
        )

    def test_without_summary_old_turns_stay_in_bridge_and_recent_is_last_three(self) -> None:
        conversation_id = self._store_conversation("no-summary")

        bundle = build_context_bundle(
            conversation_id,
            "继续讨论",
            path=self.db_path,
        )

        self.assertIsNone(bundle.rolling_summary)
        self.assertEqual(bundle.summary_revision, None)
        self.assertEqual(bundle.bridge_turn_count, 5)
        self.assertEqual(bundle.recent_turn_count, 3)
        self.assertEqual(bundle.retrieved_turn_count, 0)
        self.assertEqual(
            [turn.assistant_message_id for turn in bundle.bridge_history],
            [f"{conversation_id}-a{i}" for i in range(1, 6)],
        )
        self.assertEqual(
            [turn.assistant_message_id for turn in bundle.recent_history],
            [f"{conversation_id}-a{i}" for i in range(6, 9)],
        )

    def test_summary_bridge_recent_and_retrieved_are_disjoint(self) -> None:
        conversation_id = self._store_conversation("with-summary")
        self._summarize(conversation_id, 4)

        bundle = build_context_bundle(
            conversation_id,
            "之前凸透镜焦距怎么讨论的",
            path=self.db_path,
        )

        self.assertEqual(bundle.rolling_summary, "旧摘要")
        self.assertEqual(bundle.summary_revision, 1)
        self.assertEqual(
            [turn.assistant_message_id for turn in bundle.bridge_history],
            [f"{conversation_id}-a5"],
        )
        self.assertEqual(
            [turn.assistant_message_id for turn in bundle.recent_history],
            [f"{conversation_id}-a{i}" for i in range(6, 9)],
        )
        self.assertEqual(
            bundle.retrieved_history.turns[0].assistant_message_id,
            f"{conversation_id}-a2",
        )
        injected_ids = (
            {turn.assistant_message_id for turn in bundle.bridge_history}
            | {turn.assistant_message_id for turn in bundle.recent_history}
        )
        self.assertTrue(
            all(
                turn.assistant_message_id not in injected_ids
                for turn in bundle.retrieved_history.turns
            )
        )

    def test_summary_boundary_update_removes_bridge_on_next_build(self) -> None:
        conversation_id = self._store_conversation("boundary")
        self._summarize(conversation_id, 4, "摘要 V1")
        before = build_context_bundle(
            conversation_id, "继续", path=self.db_path
        )

        self._summarize(conversation_id, 5, "摘要 V2")
        after = build_context_bundle(
            conversation_id, "继续", path=self.db_path
        )

        self.assertEqual(before.bridge_turn_count, 1)
        self.assertEqual(after.bridge_turn_count, 0)
        self.assertEqual(after.summary_revision, 2)
        self.assertEqual(after.rolling_summary, "摘要 V2")

    def test_conversation_state_and_learning_memory_enter_bundle(self) -> None:
        conversation_id = self._store_conversation("state-memory")
        self._summarize(conversation_id, 4)
        self._add_state_and_memory(conversation_id)

        bundle = build_context_bundle(
            conversation_id,
            "继续凸透镜问题",
            path=self.db_path,
        )

        self.assertIn("凸透镜焦距与物距关系", bundle.teaching_state_context or "")
        self.assertIn("图中物体位于二倍焦距外", bundle.teaching_state_context or "")
        self.assertIn("学生曾混淆凸透镜", bundle.learning_memory_context or "")

    def test_conversation_scoped_components_are_isolated(self) -> None:
        conversation_a = self._store_conversation("A")
        contents_b = list(CONTENTS)
        contents_b[4] = ("B 独有桥接题", "B 独有桥接回答")
        conversation_b = self._store_conversation("B", contents_b)
        self._summarize(conversation_a, 4, "A 摘要")
        self._summarize(conversation_b, 4, "B 摘要")
        upsert_conversation_state(
            {
                "conversation_id": conversation_a,
                "active_problem_text": "A 独有状态",
            },
            path=self.db_path,
        )
        upsert_conversation_state(
            {
                "conversation_id": conversation_b,
                "active_problem_text": "B 独有状态",
            },
            path=self.db_path,
        )

        bundle_a = build_context_bundle(
            conversation_a, "继续", path=self.db_path
        )

        serialized = bundle_a.model_dump_json()
        self.assertIn("A 摘要", serialized)
        self.assertIn("A 独有状态", serialized)
        self.assertNotIn("B 摘要", serialized)
        self.assertNotIn("B 独有状态", serialized)
        self.assertNotIn("B 独有桥接题", serialized)
        self.assertTrue(
            all(
                turn.user_message_id.startswith(conversation_a)
                for turn in bundle_a.bridge_history + bundle_a.recent_history
            )
        )

    def test_built_bundle_is_snapshot_until_next_build(self) -> None:
        conversation_id = self._store_conversation("snapshot")
        self._summarize(conversation_id, 4, "摘要 V1")

        old_bundle = build_context_bundle(
            conversation_id, "继续", path=self.db_path
        )
        self._summarize(conversation_id, 5, "摘要 V2")

        self.assertEqual(old_bundle.rolling_summary, "摘要 V1")
        self.assertEqual(old_bundle.summary_revision, 1)
        new_bundle = build_context_bundle(
            conversation_id, "继续", path=self.db_path
        )
        self.assertEqual(new_bundle.rolling_summary, "摘要 V2")
        self.assertEqual(new_bundle.summary_revision, 2)

    def test_formatter_has_labeled_regions_and_non_authority_warning(self) -> None:
        conversation_id = self._store_conversation("formatter")
        self._summarize(conversation_id, 4)
        self._add_state_and_memory(conversation_id)
        bundle = build_context_bundle(
            conversation_id,
            "之前凸透镜焦距怎么讨论的",
            path=self.db_path,
        )

        formatted = format_context_bundle_for_model(bundle)

        for label in (
            "[Conversation Summary]",
            "[Unsummarized Recent Overflow / Bridge History]",
            "[Recent Conversation]",
            "[Retrieved Conversation History]",
            "[Current Teaching State]",
            "[Confirmed Learning Memory]",
        ):
            self.assertIn(label, formatted)
        self.assertIn("Old assistant content may be wrong", formatted)
        self.assertNotIn("trace_json", formatted)
        self.assertNotIn("tool_records", formatted)
        self.assertNotIn("data:image", formatted)

    def test_unsafe_history_is_excluded_with_visible_trim_metadata(self) -> None:
        contents = list(CONTENTS)
        contents[4] = (
            "data:image/png;base64,aGVsbG8=",
            "不应进入 Bundle",
        )
        conversation_id = self._store_conversation("unsafe", contents)
        self._summarize(conversation_id, 4)

        bundle = build_context_bundle(
            conversation_id, "继续", path=self.db_path
        )

        self.assertIn("unsafe_history", bundle.trimmed_components)
        self.assertNotIn("data:image", bundle.model_dump_json())
        self.assertEqual(bundle.bridge_turn_count, 0)

    def test_manager_does_not_call_analyzer_model_or_summary_update(self) -> None:
        source = inspect.getsource(context_manager_module).casefold()
        self.assertNotIn("analyze_question", source)
        self.assertNotIn("openai", source)
        self.assertNotIn("refresh_conversation_summary", source)
        self.assertNotIn("contextmaintenancetaskmanager", source)


if __name__ == "__main__":
    unittest.main()
