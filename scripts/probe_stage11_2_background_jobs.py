"""无 API 的 Stage 11.2 双会话后台任务人工诊断脚本。"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

# Allow direct ``python scripts/...`` execution from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation import enqueue_conversation_turn, execute_generation_job
from src.storage import (
    create_conversation,
    get_generation_job,
    initialize_database,
    list_messages,
)
from src.tasks import GenerationTaskManager


def _fake_agent_result(question: str) -> dict:
    return {
        "answer": f"Fake 后台回答：{question}",
        "analysis": {"teaching_mode": "solve"},
        "route": {
            "teaching_mode": "solve",
            "use_rag": False,
            "use_tools": False,
            "should_answer": True,
        },
        "sources": [],
        "analysis_fallback": False,
        "tool_records": [],
        "tool_model_requests": 0,
        "trace": {
            "status": "completed",
            "total_model_requests": 0,
            "total_duration_ms": 0.0,
            "run_id": "fake-background-probe",
            "steps": [],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="验证两个会话后台并发生成。")
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay 不能为负数")

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = str(Path(temp_dir) / "background_probe.db")
        initialize_database(db_path)
        conversations = [
            create_conversation("会话 A", path=db_path),
            create_conversation("会话 B", path=db_path),
        ]
        enqueued = [
            enqueue_conversation_turn(
                conversation["id"],
                f"{conversation['title']} 的 Fake 题目",
                f"{conversation['title']} 的 Fake 题目",
                db_path=db_path,
            )
            for conversation in conversations
        ]

        def slow_agent(question: str, **_kwargs) -> dict:
            time.sleep(args.delay)
            return _fake_agent_result(question)

        def execute(job_id: str, *, db_path=None):
            return execute_generation_job(
                job_id,
                db_path=db_path,
                agent_func=slow_agent,
            )

        manager = GenerationTaskManager(db_path=db_path, execute_func=execute)
        started = time.perf_counter()
        for item in enqueued:
            manager.submit(item["generation_job_id"])

        print("提交后可立即切换会话；两条 user 已持久化：")
        for conversation, item in zip(conversations, enqueued, strict=True):
            job = get_generation_job(item["generation_job_id"], path=db_path)
            print(
                f"- {conversation['title']}: status={job['status']}, "
                f"messages={len(list_messages(conversation['id'], path=db_path))}"
            )

        manager.shutdown()
        elapsed = time.perf_counter() - started
        print("后台完成：")
        for conversation, item in zip(conversations, enqueued, strict=True):
            job = get_generation_job(item["generation_job_id"], path=db_path)
            messages = list_messages(conversation["id"], path=db_path)
            print(
                f"- {conversation['title']}: status={job['status']}, "
                f"roles={[message['role'] for message in messages]}"
            )
        print(f"总耗时：{elapsed:.2f}s（单任务延迟 {args.delay:.2f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
