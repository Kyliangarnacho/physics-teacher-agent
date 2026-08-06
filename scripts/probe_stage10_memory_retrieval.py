"""Stage 10 长期记忆检索与模型注入真实探针（千问 API，临时数据库）。

A：确认机械能 misconception → 询问机械能变化，验证回答主动纠正；
B：确认多挡电路 preference → 询问养生壶多挡电路，验证结构遵循偏好；
C：deactivate 一条记忆后再次提问，验证不再注入。
只打印到 stdout，不写 Git 跟踪文件，不输出 API Key。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conversation import run_conversation_turn
from src.memory import (
    MemoryCandidate,
    confirm_memory_candidate,
    normalize_memory_content,
)
from src.storage import (
    create_conversation,
    deactivate_memory,
    initialize_database,
    list_memories,
)


def make_candidate(memory_type: str, topic: str, content: str) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=memory_type,
        topic=topic,
        content=content,
        normalized_content=normalize_memory_content(content),
        confidence=0.9,
        evidence_summary="真实探针确认",
    )


def print_turn(label: str, result: dict) -> None:
    agent_result = result["agent_result"]
    route = agent_result.get("route", {})
    print(f"--- {label} ---")
    print(
        "route: "
        f"mode={route.get('teaching_mode')} "
        f"use_rag={route.get('use_rag')} "
        f"use_tools={route.get('use_tools')}"
    )
    print(
        f"memory_count={result['memory_count']} "
        f"memory_context_used={result['memory_context_used']} "
        f"model_requests={agent_result.get('trace', {}).get('total_model_requests')}"
    )
    print(f"answer: {agent_result.get('answer', '')[:500]}")


def main() -> int:
    print("Stage 10 memory retrieval probe (real API, temp db)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "physics_teacher.db")
        initialize_database(db_path)
        conversation = create_conversation("记忆检索探针", path=db_path)
        conversation_id = conversation["id"]

        misconception = make_candidate(
            "misconception",
            "机械能守恒条件",
            "忽略空气阻力时，物体下落机械能仍守恒，不能因高度降低就判断机械能减小。",
        )
        preference = make_candidate(
            "preference",
            "多挡电路讲解偏好",
            "多挡电路先判断各挡位的串并联关系，再开始列公式。",
        )
        stored_a = confirm_memory_candidate(
            misconception,
            source_conversation_id=conversation_id,
            db_path=db_path,
        )
        stored_b = confirm_memory_candidate(
            preference,
            source_conversation_id=conversation_id,
            db_path=db_path,
        )
        print(
            f"已确认 2 条记忆: A={stored_a['memory_type']} "
            f"B={stored_b['memory_type']} "
            f"总数={len(list_memories(path=db_path))}"
        )

        try:
            result_a = run_conversation_turn(
                conversation_id,
                "忽略空气阻力时，物体从高处自由下落，机械能怎样变化？为什么不能认为机械能减小？",
                "忽略空气阻力时，物体从高处自由下落，机械能怎样变化？为什么不能认为机械能减小？",
                db_path=db_path,
            )
            print_turn("A. 机械能误区（记忆应注入）", result_a)

            result_b = run_conversation_turn(
                conversation_id,
                "养生壶有加热和保温两个挡位，对应的多挡电路怎么分析？加热挡对应电阻大还是小？",
                "养生壶有加热和保温两个挡位，对应的多挡电路怎么分析？加热挡对应电阻大还是小？",
                db_path=db_path,
            )
            print_turn("B. 多挡电路偏好（记忆应注入）", result_b)

            deactivate_memory(stored_a["id"], path=db_path)
            result_c = run_conversation_turn(
                conversation_id,
                "忽略空气阻力，小球从高处下落，机械能是否守恒？",
                "忽略空气阻力，小球从高处下落，机械能是否守恒？",
                db_path=db_path,
            )
            print_turn("C. 停用后（机械能记忆不应注入）", result_c)
        except Exception as exc:  # noqa: BLE001 - 探针需报告失败
            print(f"TURN ERROR: {type(exc).__name__}: {str(exc)[:300]}")
            return 1

        memories = list_memories(include_inactive=True, path=db_path)
        print("--- 数据库记忆 ---")
        for memory in memories:
            print(
                f"  type={memory['memory_type']} topic={memory['topic']} "
                f"confirmed={memory['confirmed']} active={memory['active']} "
                f"evidence_count={memory['evidence_count']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
