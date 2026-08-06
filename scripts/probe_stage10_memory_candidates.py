"""Stage 10 长期记忆候选与确认真实探针（千问 API，临时数据库）。

使用 A（机械能误区）、B（浮力密度）、C（电路讲解偏好）三个真实场景：
先提取候选（不写库），再手动确认一条并重复确认验证 evidence_count 递增。
只打印到 stdout，不写任何 Git 跟踪文件，不输出 API Key。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory import (
    MemoryCandidate,
    MemoryServiceError,
    confirm_memory_candidate,
    extract_memory_candidates,
)
from src.storage import initialize_database, list_memories


SCENARIOS = [
    {
        "name": "A. 机械能误区",
        "question": "为什么物体下落时机械能一定减小？",
        "answer": (
            "忽略空气阻力时，物体下落机械能守恒、机械能不变，"
            "重力势能转化为动能，所以动能增大。"
        ),
    },
    {
        "name": "B. 浮力与密度实验",
        "question": "漂浮、悬浮有什么区别？排开液体质量与物体体积是什么关系？",
        "answer": (
            "漂浮时物体部分浸入且浮力等于重力；悬浮时物体完全浸入且浮力等于重力。"
            "漂浮/悬浮时排开液体质量等于物体质量。"
        ),
    },
    {
        "name": "C. 电路讲解偏好",
        "question": "这道多挡电路题怎么分析？",
        "answer": "先判断各挡位对应电路，再代入公式计算。",
        "feedback": "以后这种多挡电路请先判断各挡位的串并联关系，再开始列公式。",
    },
]


def main() -> int:
    print("Stage 10 memory candidates probe (real API, temp db)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "physics_teacher.db")
        initialize_database(db_path)
        print(f"确认前 learning_memories 数量: {len(list_memories(path=db_path))}")

        confirmed: MemoryCandidate | None = None
        for scenario in SCENARIOS:
            print(f"--- {scenario['name']} ---")
            try:
                candidates = extract_memory_candidates(
                    scenario["question"],
                    scenario["answer"],
                    explicit_user_feedback=scenario.get("feedback"),
                )
            except Exception as exc:  # noqa: BLE001 - 探针需报告失败
                print(f"EXTRACT ERROR: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            print(f"candidates={len(candidates)}")
            for index, candidate in enumerate(candidates, start=1):
                print(
                    f"  {index}. type={candidate.memory_type} "
                    f"topic={candidate.topic} "
                    f"confidence={candidate.confidence}"
                )
                print(f"     content: {candidate.content[:120]}")
            if candidates and confirmed is None:
                confirmed = candidates[0]

        if confirmed is None:
            print("没有可确认的候选，跳过确认步骤。")
            return 0

        print("--- 手动确认（第一条候选） ---")
        try:
            first = confirm_memory_candidate(
                confirmed,
                source_conversation_id="probe-conv",
                source_message_id="probe-msg",
                db_path=db_path,
            )
            second = confirm_memory_candidate(
                confirmed,
                source_conversation_id="probe-conv",
                source_message_id="probe-msg",
                db_path=db_path,
            )
        except MemoryServiceError as exc:
            print(f"CONFIRM ERROR: {exc}")
            return 1
        print(f"第一次确认: id={first['id']} evidence_count={first['evidence_count']}")
        print(f"重复确认:   id={second['id']} evidence_count={second['evidence_count']}")
        print(f"确认后 learning_memories 数量: {len(list_memories(path=db_path))}")
        for memory in list_memories(path=db_path):
            print(
                f"  memory: type={memory['memory_type']} topic={memory['topic']} "
                f"evidence_count={memory['evidence_count']} "
                f"confirmed={memory['confirmed']} active={memory['active']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
