"""Stage 10 长期记忆候选提取与确认单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.memory import (
    MemoryCandidate,
    MemoryServiceError,
    confirm_memory_candidate,
    extract_memory_candidates,
    normalize_memory_content,
)
from src.storage import (
    initialize_database,
    list_memories,
)


def candidate_payload(
    memory_type: str = "weakness",
    topic: str = "欧姆定律",
    content: str = "串联电路计算中容易漏加总电阻",
    confidence: float = 0.8,
    evidence_summary: str = "多次出现同类错误",
) -> dict:
    return {
        "memory_type": memory_type,
        "topic": topic,
        "content": content,
        "normalized_content": content,
        "confidence": confidence,
        "evidence_summary": evidence_summary,
    }


def model_returns(payloads) -> object:
    def fake(messages):
        return json.dumps({"candidates": payloads}, ensure_ascii=False)

    return fake


class NormalizeMemoryContentTests(unittest.TestCase):
    def test_strips_and_collapses_whitespace(self) -> None:
        self.assertEqual(
            normalize_memory_content("  机械能\n  不变  "),
            "机械能 不变",
        )

    def test_fullwidth_to_halfwidth(self) -> None:
        self.assertEqual(normalize_memory_content("１２Ｖ６Ω"), "12V6Ω")
        self.assertEqual(normalize_memory_content("　全角空格　"), "全角空格")
        self.assertEqual(normalize_memory_content("中间　全角"), "中间 全角")

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(normalize_memory_content(""), "")

    def test_rejects_bytes_base64_data_url_and_secrets(self) -> None:
        with self.assertRaises(ValueError):
            normalize_memory_content(b"raw bytes")
        with self.assertRaises(ValueError):
            normalize_memory_content("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVoxMjM0NTY=")
        with self.assertRaises(ValueError):
            normalize_memory_content("data:image/png;base64,AAAA")
        with self.assertRaises(ValueError):
            normalize_memory_content("我的 DASHSCOPE_API_KEY=sk-abc123")
        with self.assertRaises(ValueError):
            normalize_memory_content("sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


class MemoryCandidateSchemaTests(unittest.TestCase):
    def test_valid_candidate_parses(self) -> None:
        candidate = MemoryCandidate.model_validate(candidate_payload())

        self.assertEqual(candidate.memory_type, "weakness")
        self.assertEqual(candidate.topic, "欧姆定律")
        self.assertGreaterEqual(candidate.confidence, 0.0)
        self.assertLessEqual(candidate.confidence, 1.0)

    def test_extra_fields_are_forbidden(self) -> None:
        payload = candidate_payload()
        payload["unexpected"] = 1
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(payload)

    def test_memory_type_only_allows_three_kinds(self) -> None:
        for memory_type in ("weakness", "misconception", "preference"):
            MemoryCandidate.model_validate(
                candidate_payload(memory_type=memory_type)
            )
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(
                candidate_payload(memory_type="mastery")
            )

    def test_confidence_must_be_between_zero_and_one(self) -> None:
        MemoryCandidate.model_validate(candidate_payload(confidence=0.0))
        MemoryCandidate.model_validate(candidate_payload(confidence=1.0))
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(candidate_payload(confidence=-0.1))
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(candidate_payload(confidence=1.1))

    def test_text_fields_reject_unsafe_content(self) -> None:
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(
                candidate_payload(content="data:image/png;base64,AAAA")
            )
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(
                candidate_payload(content="DASHSCOPE_API_KEY=sk-abc")
            )
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(
                candidate_payload(content="QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVoxMjM0NTY=")
            )
        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(candidate_payload(content="   "))


class ExtractMemoryCandidatesTests(unittest.TestCase):
    def test_parses_candidates_and_normalizes_content(self) -> None:
        payload = candidate_payload(content="  电路  串联 ")
        candidates = extract_memory_candidates(
            "问题",
            "回答",
            model_func=model_returns([payload]),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].content, "电路  串联")
        self.assertEqual(candidates[0].normalized_content, "电路 串联")

    def test_caps_at_three_candidates(self) -> None:
        payloads = [
            candidate_payload(topic=f"主题{i}")
            for i in range(5)
        ]
        candidates = extract_memory_candidates(
            "问题",
            "回答",
            model_func=model_returns(payloads),
        )

        self.assertEqual(len(candidates), 3)

    def test_invalid_json_returns_empty(self) -> None:
        candidates = extract_memory_candidates(
            "问题",
            "回答",
            model_func=lambda messages: "这不是 JSON",
        )
        self.assertEqual(candidates, [])

    def test_non_candidates_payload_returns_empty(self) -> None:
        candidates = extract_memory_candidates(
            "问题",
            "回答",
            model_func=lambda messages: json.dumps({"foo": 1}),
        )
        self.assertEqual(candidates, [])

    def test_invalid_entries_are_skipped(self) -> None:
        good = candidate_payload()
        bad = candidate_payload(topic="坏")
        bad["unexpected"] = 1
        candidates = extract_memory_candidates(
            "问题",
            "回答",
            model_func=model_returns([good, bad]),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].topic, "欧姆定律")

    def test_system_problem_returns_empty_without_calling_model(self) -> None:
        def should_not_be_called(messages):
            raise AssertionError("系统问题不应调用模型")

        candidates = extract_memory_candidates(
            "电压 12V 求电流",
            "工具参数校验失败，请检查必填字段。",
            model_func=should_not_be_called,
        )
        self.assertEqual(candidates, [])

        candidates = extract_memory_candidates(
            "图中电压表读数是多少？",
            "请补充题图，或完整描述图中的物体、连接关系和已知信息。",
            model_func=should_not_be_called,
        )
        self.assertEqual(candidates, [])

    def test_model_exception_returns_empty(self) -> None:
        def broken(messages):
            raise RuntimeError("boom")

        candidates = extract_memory_candidates(
            "问题",
            "回答",
            model_func=broken,
        )
        self.assertEqual(candidates, [])

    def test_invalid_conversation_history_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_memory_candidates(
                "问题",
                "回答",
                conversation_history=[{"role": "system", "content": "x"}],
                model_func=model_returns([]),
            )

    def test_empty_question_or_answer_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_memory_candidates("", "回答", model_func=model_returns([]))
        with self.assertRaises(ValueError):
            extract_memory_candidates("问题", "  ", model_func=model_returns([]))

    def test_extractor_does_not_write_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "memory.db")
            initialize_database(db_path)
            extract_memory_candidates(
                "问题",
                "回答",
                model_func=model_returns([candidate_payload()]),
            )
            self.assertEqual(list_memories(path=db_path), [])


class ExamScenarioTests(unittest.TestCase):
    """基于真实中考场景的候选提取边界。"""

    def test_scenario_a_mechanical_energy_misconception(self) -> None:
        question = "为什么物体下落时机械能一定减小？"
        answer = (
            "忽略空气阻力时，物体下落机械能守恒、机械能不变，"
            "重力势能转化为动能，所以动能增大。"
        )
        payload = candidate_payload(
            memory_type="misconception",
            topic="机械能守恒",
            content="把“有空气阻力时机械能减小”误认为下落时机械能一定减小",
            confidence=0.85,
        )

        candidates = extract_memory_candidates(
            question,
            answer,
            model_func=model_returns([payload]),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "misconception")
        self.assertIn("机械能", candidates[0].topic)

    def test_scenario_b_buoyancy_density_weakness(self) -> None:
        question = "漂浮、悬浮有什么区别？排开液体质量与物体体积是什么关系？"
        answer = (
            "漂浮时物体部分浸入且浮力等于重力；悬浮时物体完全浸入且浮力等于重力。"
            "排开液体质量等于物体质量（漂浮/悬浮时）。"
        )
        payload = candidate_payload(
            memory_type="weakness",
            topic="浮力与密度",
            content="反复混淆漂浮、悬浮以及排开液体质量和物体体积的关系",
            confidence=0.8,
        )

        candidates = extract_memory_candidates(
            question,
            answer,
            model_func=model_returns([payload]),
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn(candidates[0].memory_type, ("weakness", "misconception"))
        self.assertIn("浮力", candidates[0].topic)

    def test_scenario_c_circuit_preference(self) -> None:
        question = "这道多挡电路题怎么分析？"
        answer = "先判断各挡位对应电路，再代入公式计算。"
        feedback = "以后这种多挡电路请先判断各挡位的串并联关系，再开始列公式。"
        payload = candidate_payload(
            memory_type="preference",
            topic="多挡电路讲解偏好",
            content="多挡电路先判断各挡位的串并联关系，再列公式",
            confidence=0.95,
        )

        candidates = extract_memory_candidates(
            question,
            answer,
            explicit_user_feedback=feedback,
            model_func=model_returns([payload]),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "preference")
        self.assertIn("串并联", candidates[0].content)

    def test_scenario_d_correct_answer_without_preference_no_memory(self) -> None:
        question = "选择：下列做法符合安全用电原则的是？"
        answer = "正确答案是 B：不接触低压带电体，不靠近高压带电体。"

        candidates = extract_memory_candidates(
            question,
            answer,
            model_func=model_returns([]),
        )

        self.assertEqual(candidates, [])


class ConfirmMemoryCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "memory.db")
        initialize_database(self.db_path)

    def test_confirm_writes_confirmed_active_memory(self) -> None:
        candidate = MemoryCandidate.model_validate(candidate_payload())

        stored = confirm_memory_candidate(
            candidate,
            source_conversation_id="conv-1",
            source_message_id="msg-1",
            db_path=self.db_path,
        )

        self.assertEqual(stored["confirmed"], 1)
        self.assertEqual(stored["active"], 1)
        self.assertEqual(stored["evidence_count"], 1)
        self.assertEqual(stored["source_conversation_id"], "conv-1")
        self.assertEqual(stored["source_message_id"], "msg-1")
        self.assertNotIn("evidence_summary", stored)
        memories = list_memories(path=self.db_path)
        self.assertEqual(len(memories), 1)

    def test_repeat_confirm_increments_evidence_count(self) -> None:
        candidate = MemoryCandidate.model_validate(candidate_payload())
        first = confirm_memory_candidate(candidate, db_path=self.db_path)
        second = confirm_memory_candidate(candidate, db_path=self.db_path)

        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["evidence_count"], 2)
        self.assertEqual(len(list_memories(path=self.db_path)), 1)

    def test_confirm_accepts_dict(self) -> None:
        stored = confirm_memory_candidate(
            candidate_payload(),
            db_path=self.db_path,
        )
        self.assertEqual(stored["confirmed"], 1)

    def test_invalid_candidate_raises_service_error(self) -> None:
        with self.assertRaises(MemoryServiceError):
            confirm_memory_candidate(
                {"memory_type": "mastery", "topic": "x"},
                db_path=self.db_path,
            )


if __name__ == "__main__":
    unittest.main()
