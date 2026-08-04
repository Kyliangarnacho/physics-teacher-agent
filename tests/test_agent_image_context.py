import json
import unittest

from src.agent import run_teacher_agent
from tests.test_agent import (
    RAG_CARDS,
    FakeAnalyzer,
    FakeAnswer,
    FakeRetriever,
    FakeToolAnswer,
    analysis_json,
)


CONFIRMED_CONTEXT = "图中电源电压为 12 V，电阻为 6 Ω，电阻与电流表串联。"


def failed_tool_result(error_type: str = "tool_validation") -> dict:
    return {
        "answer": "工具参数校验失败。",
        "tool_records": [
            {
                "tool_call_id": "call-failed",
                "name": "calculate_electric_power",
                "status": "error",
            }
        ],
        "model_requests": 1,
        "step_traces": [
            {
                "name": "tool_selection",
                "status": "success",
                "attempts": 1,
                "duration_ms": 1,
                "model_requests": 1,
            },
            {
                "name": "tool_execution",
                "status": "error",
                "attempts": 1,
                "duration_ms": 1,
                "model_requests": 0,
                "error_type": error_type,
                "error_message": "本地工具参数校验失败。",
            },
            {
                "name": "tool_result_answer",
                "status": "skipped",
                "attempts": 0,
                "duration_ms": 0,
                "model_requests": 0,
            },
        ],
    }


class AgentImageContextTests(unittest.TestCase):
    def test_image_tool_validation_falls_back_to_plain_answer(self):
        answer = FakeAnswer("已按图片 1、图片 2 分别回答。")
        image_context = "【图片 1】计算图中用电器的电功率。"

        result = run_teacher_agent(
            "请计算图片中的电功率。",
            image_context=image_context,
            image_context_available=True,
            analyzer_func=FakeAnalyzer(
                analysis_json(calculation_required=True, teaching_mode="solve")
            ),
            answer_func=answer,
            tool_answer_func=lambda *_args, **_kwargs: failed_tool_result(),
        )

        self.assertEqual(result["answer"], "已按图片 1、图片 2 分别回答。")
        self.assertEqual(len(answer.calls), 1)
        self.assertEqual(result["trace"]["status"], "completed_with_fallback")
        fallback_step = result["trace"]["steps"][-1]
        self.assertEqual(fallback_step["name"], "final_answer")
        self.assertTrue(fallback_step["metadata"]["tool_fallback"])
        self.assertEqual(fallback_step["metadata"]["reason"], "tool_validation")
        self.assertNotIn(image_context, json.dumps(result["trace"], ensure_ascii=False))

    def test_explicit_separate_multi_image_questions_skip_single_tool(self):
        answer = FakeAnswer("图1和图2已分别完整回答。")
        tool_answer = FakeToolAnswer()
        multi_context = (
            "【多图回答规则】\n多张图片若是独立题目，请按图片编号分别回答。\n\n"
            "【图片 1：one.png】\n第一题\n\n【图片 2：two.png】\n第二题"
        )

        result = run_teacher_agent(
            "请依次解答这两张图片。",
            image_context=multi_context,
            image_context_available=True,
            analyzer_func=FakeAnalyzer(
                analysis_json(calculation_required=True, teaching_mode="solve")
            ),
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertEqual(tool_answer.calls, [])
        self.assertEqual(len(answer.calls), 1)
        self.assertEqual(result["answer"], "图1和图2已分别完整回答。")
        self.assertEqual(result["trace"]["status"], "completed_with_fallback")
        self.assertEqual(
            result["trace"]["steps"][-1]["metadata"]["reason"],
            "no_single_tool_for_multiple_image_questions",
        )

    def test_successful_image_tool_path_does_not_fallback(self):
        answer = FakeAnswer()
        tool_answer = FakeToolAnswer()

        result = run_teacher_agent(
            "求电流。",
            image_context=CONFIRMED_CONTEXT,
            image_context_available=True,
            analyzer_func=FakeAnalyzer(
                analysis_json(calculation_required=True, teaching_mode="solve")
            ),
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertEqual(result["answer"], "工具教师回答")
        self.assertEqual(answer.calls, [])
        self.assertFalse(
            any(
                step.get("metadata", {}).get("tool_fallback")
                for step in result["trace"]["steps"]
            )
        )

    def test_plain_text_tool_validation_keeps_existing_error_behavior(self):
        answer = FakeAnswer()

        result = run_teacher_agent(
            "计算一道电功率题。",
            analyzer_func=FakeAnalyzer(
                analysis_json(calculation_required=True, teaching_mode="solve")
            ),
            answer_func=answer,
            tool_answer_func=lambda *_args, **_kwargs: failed_tool_result(),
        )

        self.assertEqual(result["answer"], "工具参数校验失败。")
        self.assertEqual(answer.calls, [])
        self.assertEqual(result["trace"]["status"], "completed")

    def test_image_required_without_context_remains_blocked(self):
        answer = FakeAnswer()

        result = run_teacher_agent(
            "请根据图片回答。",
            analyzer_func=FakeAnalyzer(analysis_json(image_required=True)),
            answer_func=answer,
        )

        self.assertFalse(result["route"]["should_answer"])
        self.assertEqual(answer.calls, [])

    def test_confirmed_context_allows_image_required_question(self):
        answer = FakeAnswer()

        result = run_teacher_agent(
            "请根据图片回答。",
            image_context=CONFIRMED_CONTEXT,
            image_context_available=True,
            analyzer_func=FakeAnalyzer(analysis_json(image_required=True)),
            answer_func=answer,
        )

        self.assertTrue(result["route"]["should_answer"])
        self.assertEqual(len(answer.calls), 1)

    def test_available_context_must_be_nonempty_string(self):
        analyzer = FakeAnalyzer(analysis_json())

        for context in (None, "", "   "):
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, "image_context 不能为空"):
                    run_teacher_agent(
                        "请看图回答。",
                        image_context=context,
                        image_context_available=True,
                        analyzer_func=analyzer,
                    )

        self.assertEqual(analyzer.calls, [])

    def test_unconfirmed_context_is_ignored_everywhere(self):
        analyzer = FakeAnalyzer(analysis_json())
        answer = FakeAnswer()
        unconfirmed = "绝不能使用的未确认文字"

        run_teacher_agent(
            "解释浮力。",
            image_context=unconfirmed,
            image_context_available=False,
            analyzer_func=analyzer,
            answer_func=answer,
        )

        self.assertEqual(analyzer.calls, ["解释浮力。"])
        self.assertEqual(answer.calls[0]["question"], "解释浮力。")
        self.assertNotIn(unconfirmed, analyzer.calls[0])

    def test_confirmed_image_text_enters_analyzer(self):
        analyzer = FakeAnalyzer(analysis_json())

        run_teacher_agent(
            "求电流。",
            image_context=CONFIRMED_CONTEXT,
            image_context_available=True,
            analyzer_func=analyzer,
            answer_func=FakeAnswer(),
        )

        analyzed = analyzer.calls[0]
        self.assertEqual(
            analyzed,
            "用户要求：\n求电流。\n\n已确认图片上下文：\n" + CONFIRMED_CONTEXT,
        )

    def test_graph_relationship_enters_plain_answer(self):
        relationship = "s-t 图像是一条经过原点的直线，10 s 时路程为 50 m。"
        answer = FakeAnswer()

        run_teacher_agent(
            "说明图像表示的运动状态。",
            image_context=relationship,
            image_context_available=True,
            analyzer_func=FakeAnalyzer(analysis_json(image_required=True)),
            answer_func=answer,
        )

        self.assertIn("已确认图片上下文", answer.calls[0]["question"])
        self.assertIn(relationship, answer.calls[0]["question"])

    def test_confirmed_image_context_enters_rag_query(self):
        retriever = FakeRetriever(RAG_CARDS)

        run_teacher_agent(
            "结合资料解释。",
            image_context=CONFIRMED_CONTEXT,
            image_context_available=True,
            rag_policy="force",
            analyzer_func=FakeAnalyzer(analysis_json(image_required=True)),
            retriever=retriever,
            answer_func=FakeAnswer(),
        )

        query, top_k = retriever.calls[0]
        self.assertIn(CONFIRMED_CONTEXT, query)
        self.assertEqual(top_k, 3)

    def test_confirmed_image_calculation_can_use_tool_path(self):
        answer = FakeAnswer()
        tool_answer = FakeToolAnswer()

        result = run_teacher_agent(
            "求通过电阻的电流。",
            image_context=CONFIRMED_CONTEXT,
            image_context_available=True,
            rag_policy="off",
            analyzer_func=FakeAnalyzer(
                analysis_json(
                    image_required=True,
                    calculation_required=True,
                    teaching_mode="solve",
                )
            ),
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertTrue(result["route"]["use_tools"])
        self.assertEqual(answer.calls, [])
        self.assertIn(CONFIRMED_CONTEXT, tool_answer.calls[0]["question"])

    def test_image_concept_question_does_not_force_tools(self):
        tool_answer = FakeToolAnswer()
        answer = FakeAnswer()

        result = run_teacher_agent(
            "解释图中现象。",
            image_context="图中金属块与木块温度相同。",
            image_context_available=True,
            analyzer_func=FakeAnalyzer(
                analysis_json(
                    image_required=True,
                    calculation_required=False,
                    teaching_mode="explain",
                )
            ),
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertFalse(result["route"]["use_tools"])
        self.assertEqual(len(answer.calls), 1)
        self.assertEqual(tool_answer.calls, [])

    def test_legacy_text_path_keeps_original_question(self):
        analyzer = FakeAnalyzer(analysis_json())
        answer = FakeAnswer()

        run_teacher_agent(
            "为什么金属摸起来更凉？",
            analyzer_func=analyzer,
            answer_func=answer,
        )

        self.assertEqual(analyzer.calls, ["为什么金属摸起来更凉？"])
        self.assertEqual(
            answer.calls[0]["question"],
            "为什么金属摸起来更凉？",
        )

    def test_trace_does_not_contain_image_context_or_base64(self):
        sensitive_context = (
            "图中关系标记 UNIQUE_IMAGE_CONTEXT，原图内容为 "
            "data:image/png;base64,SHOULD_NOT_APPEAR。"
        )

        result = run_teacher_agent(
            "解释图中关系。",
            image_context=sensitive_context,
            image_context_available=True,
            analyzer_func=FakeAnalyzer(analysis_json(image_required=True)),
            answer_func=FakeAnswer(),
        )
        serialized_trace = json.dumps(result["trace"], ensure_ascii=False)

        self.assertNotIn("UNIQUE_IMAGE_CONTEXT", serialized_trace)
        self.assertNotIn("data:image", serialized_trace)
        self.assertNotIn("base64", serialized_trace)


if __name__ == "__main__":
    unittest.main()
