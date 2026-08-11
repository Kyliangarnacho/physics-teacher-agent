"""Conversation-scoped local BM25 history retrieval tests."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import src.context.history_retrieval as history_retrieval_module
from src.context.history_retrieval import (
    DEFAULT_HISTORY_MIN_ABSOLUTE_SCORE,
    RETRIEVED_HISTORY_CAUTION,
    format_retrieved_history,
    retrieve_conversation_history,
    retrieve_history_from_records,
    tokenize_history_text,
)
from src.storage import (
    create_conversation,
    initialize_database,
    insert_message,
    upsert_conversation_summary,
)


def _messages(contents: list[tuple[str, str]]) -> list[dict]:
    records: list[dict] = []
    for index, (user_content, assistant_content) in enumerate(contents, 1):
        records.extend(
            [
                {
                    "id": f"u{index}",
                    "conversation_id": "conv-1",
                    "role": "user",
                    "display_content": user_content,
                    "model_content": user_content,
                    "image_metadata_json": [],
                    "created_at": f"2026-08-10T00:{index:02d}:00+00:00",
                },
                {
                    "id": f"a{index}",
                    "conversation_id": "conv-1",
                    "role": "assistant",
                    "display_content": assistant_content,
                    "model_content": assistant_content,
                    "image_metadata_json": [],
                    "created_at": f"2026-08-10T00:{index:02d}:01+00:00",
                },
            ]
        )
    return records


def _summary(boundary: str, covered_turn_count: int) -> dict:
    return {
        "conversation_id": "conv-1",
        "summary_text": "旧对话摘要",
        "covered_until_message_id": boundary,
        "covered_turn_count": covered_turn_count,
        "summary_revision": 1,
        "model_name": "qwen-test",
        "created_at": "2026-08-10T01:00:00+00:00",
        "updated_at": "2026-08-10T01:00:00+00:00",
    }


BASE_CONTENTS = [
    ("我们讨论过并联电路电阻变化", "旧回答说支路电流会发生变化"),
    ("凸透镜焦距和成像位置", "旧回答比较了物距与焦距"),
    ("浮力与排开液体体积", "旧回答使用阿基米德原理"),
    ("弹簧振子共振频率", "旧回答讨论固有频率"),
    ("斜面机械效率实验", "旧回答记录了拉力和斜面长度"),
    ("桥接轮次中的天平校准", "尚未被摘要覆盖的旧回答"),
    ("桥接轮次中的滑轮组", "尚未被摘要覆盖的另一个回答"),
    ("最近轮次中的声速", "最近回答一"),
    ("最近轮次中的光的折射", "最近回答二"),
    ("最近轮次中的电磁铁", "最近回答三"),
]


class HistoryRetrievalPureTests(unittest.TestCase):
    def test_chinese_jieba_tokenization(self) -> None:
        tokens = tokenize_history_text("凸透镜成像时焦距如何变化")

        self.assertIn("凸透镜", tokens)
        self.assertIn("焦距", tokens)

    def test_covered_old_turn_is_retrieved_but_recent_and_bridge_are_excluded(self) -> None:
        messages = _messages(BASE_CONTENTS)
        summary = _summary("a5", 5)

        covered = retrieve_history_from_records(
            messages, [], summary, "之前的凸透镜焦距结论"
        )
        bridge = retrieve_history_from_records(
            messages, [], summary, "天平校准"
        )
        recent = retrieve_history_from_records(
            messages, [], summary, "电磁铁"
        )

        self.assertEqual(covered.candidate_count, 5)
        self.assertEqual(covered.turns[0].assistant_message_id, "a2")
        self.assertEqual(bridge.turns, [])
        self.assertEqual(recent.turns, [])

    def test_without_summary_all_old_turns_are_bridge_and_no_candidates_remain(self) -> None:
        result = retrieve_history_from_records(
            _messages(BASE_CONTENTS), [], None, "凸透镜焦距"
        )

        self.assertEqual(result.turns, [])
        self.assertEqual(result.candidate_count, 0)
        self.assertFalse(result.retrieval_used)

    def test_low_relevance_returns_empty(self) -> None:
        result = retrieve_history_from_records(
            _messages(BASE_CONTENTS),
            [],
            _summary("a5", 5),
            "企鹅在南极怎样孵蛋",
        )

        self.assertEqual(result.turns, [])
        self.assertTrue(result.retrieval_used)

    def test_first_weak_positive_match_does_not_bypass_absolute_floor(self) -> None:
        result = retrieve_history_from_records(
            _messages(BASE_CONTENTS),
            [],
            _summary("a5", 5),
            "\u7269\u4f53\u600e\u6837\u53d8\u5316",
        )

        self.assertEqual(result.turns, [])
        self.assertTrue(result.retrieval_used)

    def test_one_clear_old_topic_match_survives_absolute_floor(self) -> None:
        result = retrieve_history_from_records(
            _messages(BASE_CONTENTS),
            [],
            _summary("a5", 5),
            "\u51f8\u900f\u955c\u7126\u8ddd\u7ed3\u8bba",
        )

        self.assertEqual([turn.assistant_message_id for turn in result.turns], ["a2"])
        self.assertGreaterEqual(result.turns[0].score, DEFAULT_HISTORY_MIN_ABSOLUTE_SCORE)

    def test_default_top_k_is_two_scores_descend_and_newer_breaks_tie(self) -> None:
        contents = [
            ("共振实验相同内容", "共振实验相同回答"),
            ("共振实验相同内容", "共振实验相同回答"),
            ("共振实验相同内容", "共振实验相同回答"),
            ("无关旧题", "无关旧回答"),
            ("无关旧题二", "无关旧回答二"),
            ("最近一", "最近回答一"),
            ("最近二", "最近回答二"),
            ("最近三", "最近回答三"),
        ]
        result = retrieve_history_from_records(
            _messages(contents), [], _summary("a5", 5), "共振实验"
        )

        self.assertEqual(len(result.turns), 2)
        self.assertEqual(
            [turn.assistant_message_id for turn in result.turns],
            ["a3", "a2"],
        )
        scores = [turn.score for turn in result.turns]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(
            all(score >= DEFAULT_HISTORY_MIN_ABSOLUTE_SCORE for score in scores)
        )

    def test_recent_topic_does_not_pull_weak_or_unrelated_covered_history(self) -> None:
        contents = [
            ("\u65e7\u9898\u76ee\u5149\u5b66", "\u65e7\u56de\u7b54\u955c\u9762\u53cd\u5c04"),
            ("\u65e7\u9898\u76ee\u7535\u5b66", "\u65e7\u56de\u7b54\u4e32\u8054\u7535\u8def"),
            ("\u65e7\u9898\u76ee\u58f0\u5b66", "\u65e7\u56de\u7b54\u58f0\u97f3\u4f20\u64ad"),
            ("\u65e7\u9898\u76ee\u70ed\u5b66", "\u65e7\u56de\u7b54\u5185\u80fd\u53d8\u5316"),
            ("\u65e7\u9898\u76ee\u8fd0\u52a8", "\u65e7\u56de\u7b54\u901f\u5ea6\u53d8\u5316"),
            ("\u6d6e\u529b\u6700\u8fd1\u9898", "\u6700\u8fd1\u6d6e\u529b\u56de\u7b54"),
            ("\u6d6e\u529b\u7ee7\u7eed\u8ffd\u95ee", "\u6700\u8fd1\u6d6e\u529b\u6761\u4ef6"),
            ("\u6d6e\u529b\u7ed3\u8bba", "\u6700\u8fd1\u6d6e\u529b\u89e3\u91ca"),
        ]
        result = retrieve_history_from_records(
            _messages(contents), [], _summary("a5", 5), "\u6d6e\u529b\u600e\u6837\u8ba1\u7b97"
        )

        self.assertEqual(result.candidate_count, 5)
        self.assertEqual(result.turns, [])

    def test_real_buoyancy_follow_up_ignores_generic_old_option_words(self) -> None:
        contents = [
            (
                "换一道题：汽车安全带为什么要设计得比较宽？A 或 B，选一个并解释。",
                "选 B。面积大时压强小，但旧回答只代表当时的解释。",
            ),
            (
                "那为什么 A 不对？人受到的力会不会变小？",
                "力不因安全带面积直接改变，这一点要区分。",
            ),
            (
                "视频快速播放时，A 音量还是 B 音调变化？",
                "选择 B，只说明音调变化。",
            ),
            ("木块速度为什么没用？", "匀速时利用二力平衡。"),
            ("凸透镜焦距问题", "讨论物距和像距。"),
            ("弹簧振子周期", "讨论周期条件。"),
            ("木块压力是多少？", "根据重力得到压力。"),
            ("串联电路电源电压", "列方程得到电压。"),
            ("变阻器最小阻值", "串联电路继续计算。"),
            (
                "同种液体中，体积相等的 a、b，a 漂浮、b 悬浮。",
                "最近一轮讨论浮力条件。",
            ),
        ]
        result = retrieve_history_from_records(
            _messages(contents),
            [],
            _summary("a7", 7),
            "那为什么同样体积时 a 受到的浮力反而比 b 小？只解释。",
            active_problem_text=contents[-1][0],
        )

        self.assertEqual(result.candidate_count, 7)
        self.assertEqual(result.turns, [])
        self.assertNotIn("a", tokenize_history_text(result.query))
        self.assertNotIn("b", tokenize_history_text(result.query))
        self.assertNotIn("的", tokenize_history_text(result.query))

    def test_old_circuit_turn_still_has_real_topic_recall(self) -> None:
        contents = [
            (
                "串联电路中电流 0.2 A 时变阻器电压 10 V，求电源电压。",
                "由串联电路和欧姆定律得到电源电压 12 V。",
            ),
            ("凸透镜成像", "讨论焦距。"),
            ("声音传播", "讨论音调。"),
            ("木块摩擦", "讨论二力平衡。"),
            ("弹簧振动", "讨论周期。"),
            ("最近浮力一", "最近回答一。"),
            ("最近浮力二", "最近回答二。"),
            ("最近浮力三", "最近回答三。"),
        ]
        result = retrieve_history_from_records(
            _messages(contents),
            [],
            _summary("a5", 5),
            "之前串联电路算出的电源电压是多少？",
        )

        self.assertGreaterEqual(len(result.turns), 1)
        self.assertEqual(result.turns[0].assistant_message_id, "a1")

    def test_active_problem_text_supplements_query_and_blank_is_ignored(self) -> None:
        messages = _messages(BASE_CONTENTS)
        summary = _summary("a5", 5)

        supplemented = retrieve_history_from_records(
            messages,
            [],
            summary,
            "继续刚才内容",
            active_problem_text="凸透镜焦距和物距关系",
        )
        blank = retrieve_history_from_records(
            messages,
            [],
            summary,
            "凸透镜焦距",
            active_problem_text="   ",
        )

        self.assertEqual(supplemented.turns[0].assistant_message_id, "a2")
        self.assertEqual(
            supplemented.query,
            "继续刚才内容\n凸透镜焦距和物距关系",
        )
        self.assertEqual(blank.query, "凸透镜焦距")
        self.assertEqual(blank.turns[0].assistant_message_id, "a2")

    def test_current_and_isolated_user_do_not_form_candidates(self) -> None:
        messages = _messages(BASE_CONTENTS)
        messages.append(
            {
                "id": "u11",
                "conversation_id": "conv-1",
                "role": "user",
                "display_content": "孤立用户独有词",
                "model_content": "孤立用户独有词",
                "image_metadata_json": [],
                "created_at": "2026-08-10T00:11:00+00:00",
            }
        )
        result = retrieve_history_from_records(
            messages, [], _summary("a5", 5), "孤立用户独有词"
        )

        self.assertEqual(result.turns, [])
        self.assertEqual(result.candidate_count, 5)

    def test_non_completed_jobs_do_not_form_candidates(self) -> None:
        for status in ("pending", "running", "failed", "interrupted"):
            with self.subTest(status=status):
                result = retrieve_history_from_records(
                    _messages(BASE_CONTENTS[:8]),
                    [
                        {
                            "user_message_id": "u1",
                            "status": status,
                            "payload": {},
                        }
                    ],
                    _summary("a5", 5),
                    "并联电路电阻",
                )
                self.assertEqual(result.turns, [])
                self.assertEqual(result.candidate_count, 4)

    def test_unsafe_candidate_is_excluded_and_internal_records_are_not_exposed(self) -> None:
        contents = list(BASE_CONTENTS[:8])
        contents[0] = (
            "data:image/png;base64,aGVsbG8=",
            "tool_records trace_json protocol",
        )
        unsafe_result = retrieve_history_from_records(
            _messages(contents), [], _summary("a5", 5), "tool records protocol"
        )
        safe_result = retrieve_history_from_records(
            _messages(contents), [], _summary("a5", 5), "凸透镜焦距"
        )

        self.assertEqual(unsafe_result.candidate_count, 4)
        self.assertEqual(unsafe_result.turns, [])
        self.assertEqual(safe_result.turns[0].assistant_message_id, "a2")
        allowed_fields = {
            "user_message_id",
            "assistant_message_id",
            "user_content",
            "assistant_content",
            "score",
        }
        self.assertTrue(
            all(
                set(turn.model_dump()) == allowed_fields
                for turn in safe_result.turns
            )
        )

    def test_formatter_marks_old_assistant_as_non_authoritative(self) -> None:
        result = retrieve_history_from_records(
            _messages(BASE_CONTENTS),
            [],
            _summary("a5", 5),
            "凸透镜焦距",
        )
        formatted = format_retrieved_history(result)

        self.assertIn(RETRIEVED_HISTORY_CAUTION, formatted)
        self.assertIn("Old assistant content may be wrong", formatted)
        self.assertIn("reliable physics knowledge or RAG take precedence", formatted)

    def test_module_has_no_qwen_openai_or_physics_rag_call(self) -> None:
        source = inspect.getsource(history_retrieval_module).casefold()
        self.assertNotIn("openai", source)
        self.assertNotIn("qwen", source)
        self.assertNotIn("knowledgeretriever", source)


class ConversationScopedHistoryRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "history-retrieval.db")
        initialize_database(self.db_path)

    def _store_conversation(
        self,
        title: str,
        contents: list[tuple[str, str]],
        covered_turn_count: int,
    ) -> str:
        conversation_id = create_conversation(title, path=self.db_path)["id"]
        for index, (user_content, assistant_content) in enumerate(contents, 1):
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
        upsert_conversation_summary(
            conversation_id,
            f"{title} summary",
            covered_until_message_id=f"{conversation_id}-a{covered_turn_count}",
            covered_turn_count=covered_turn_count,
            model_name="qwen-test",
            path=self.db_path,
        )
        return conversation_id

    def test_repository_wrapper_searches_only_current_conversation(self) -> None:
        conversation_a = self._store_conversation("A", BASE_CONTENTS[:8], 5)
        contents_b = list(BASE_CONTENTS[:8])
        contents_b[1] = ("月球车独有悬架结构", "旧回答讨论六轮悬架")
        conversation_b = self._store_conversation("B", contents_b, 5)

        result_a = retrieve_conversation_history(
            conversation_a, "月球车独有悬架", path=self.db_path
        )
        result_b = retrieve_conversation_history(
            conversation_b, "月球车独有悬架", path=self.db_path
        )

        self.assertEqual(result_a.turns, [])
        self.assertEqual(
            result_b.turns[0].assistant_message_id,
            f"{conversation_b}-a2",
        )
        self.assertTrue(
            all(
                turn.user_message_id.startswith(conversation_b)
                for turn in result_b.turns
            )
        )

    def test_identical_content_in_two_conversations_returns_local_message_ids(self) -> None:
        conversation_a = self._store_conversation("same-A", BASE_CONTENTS[:8], 5)
        conversation_b = self._store_conversation("same-B", BASE_CONTENTS[:8], 5)

        result_a = retrieve_conversation_history(
            conversation_a, "凸透镜焦距", path=self.db_path
        )
        result_b = retrieve_conversation_history(
            conversation_b, "凸透镜焦距", path=self.db_path
        )

        self.assertTrue(result_a.turns)
        self.assertTrue(result_b.turns)
        self.assertTrue(
            all(
                turn.user_message_id.startswith(conversation_a)
                for turn in result_a.turns
            )
        )
        self.assertTrue(
            all(
                turn.user_message_id.startswith(conversation_b)
                for turn in result_b.turns
            )
        )


if __name__ == "__main__":
    unittest.main()
