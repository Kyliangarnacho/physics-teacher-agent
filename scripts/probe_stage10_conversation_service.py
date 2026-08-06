"""Stage 10 Conversation Service 真实两轮探针（临时数据库，千问 API）。

流程：同一会话先问“12 V、6 Ω，求电流，只给第一步”，再发“继续下一步”，
验证第二轮能读取第一轮历史与状态、hint_step 递增、回答承接原题。
临时数据库位于系统临时目录，不进入 Git；只打印，不写任何 Git 跟踪文件。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conversation import ConversationServiceError, run_conversation_turn
from src.agent import run_teacher_agent
from src.storage import (
    create_conversation,
    get_agent_runs,
    get_conversation_state,
    initialize_database,
    list_messages,
)


ROUND_ONE_QUESTION = "12 V、6 Ω，求电流，只给第一步，不要直接给最终答案"
ROUND_TWO_QUESTION = "继续下一步"


def print_round(label: str, result: dict) -> None:
    agent_result = result["agent_result"]
    route = agent_result.get("route", {})
    trace = agent_result.get("trace", {})
    print(f"--- {label} ---")
    print(f"question(agent): {result['agent_result'].get('_question', '')}")
    print(
        "route: "
        f"mode={route.get('teaching_mode')} "
        f"use_rag={route.get('use_rag')} "
        f"use_tools={route.get('use_tools')} "
        f"should_answer={route.get('should_answer')}"
    )
    print(
        "trace: "
        f"status={trace.get('status')} "
        f"total_model_requests={trace.get('total_model_requests')} "
        f"tool_executions={trace.get('tool_executions')}"
    )
    print(f"history_turn_count={result['history_turn_count']} "
          f"state_context_used={result['state_context_used']}")
    print(f"answer: {agent_result.get('answer', '')[:600]}")
    for record in agent_result.get("tool_records", []) or []:
        print(
            f"tool_record: name={record.get('name')} "
            f"status={record.get('status')} "
            f"arguments={record.get('arguments')} "
            f"error={record.get('error')}"
        )


def main() -> int:
    print("Stage 10 conversation service probe (real API, temp db)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "physics_teacher.db")
        initialize_database(db_path)
        conversation = create_conversation("Stage10 两轮探针", path=db_path)
        conversation_id = conversation["id"]
        agent_questions: list[str] = []

        def recording_agent(question: str, **kwargs) -> dict:
            agent_questions.append(question)
            return run_teacher_agent(question, **kwargs)

        try:
            round_one = run_conversation_turn(
                conversation_id,
                display_question=ROUND_ONE_QUESTION,
                model_question=ROUND_ONE_QUESTION,
                mode_override="hint",
                db_path=db_path,
                agent_func=recording_agent,
            )
        except ConversationServiceError as exc:
            print(f"ROUND 1 ERROR: {exc}")
            return 1
        round_one["agent_result"]["_question"] = agent_questions[-1]
        print_round("第一轮", round_one)

        try:
            round_two = run_conversation_turn(
                conversation_id,
                display_question=ROUND_TWO_QUESTION,
                model_question=ROUND_TWO_QUESTION,
                db_path=db_path,
                agent_func=recording_agent,
            )
        except ConversationServiceError as exc:
            print(f"ROUND 2 ERROR: {exc}")
            return 1
        round_two["agent_result"]["_question"] = agent_questions[-1]
        print_round("第二轮", round_two)

        messages = list_messages(conversation_id, path=db_path)
        runs = get_agent_runs(conversation_id, path=db_path)
        state = get_conversation_state(conversation_id, path=db_path)
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        print("--- 数据库记录 ---")
        print(f"messages={len(messages)} (user={len(user_messages)}, "
              f"assistant={len(assistant_messages)})")
        print(f"agent_runs={len(runs)} statuses={[r['status'] for r in runs]}")
        print(f"conversation_state={state}")

        second_answer = round_two["agent_result"].get("answer", "")
        tool_text = str(round_two["agent_result"].get("tool_records", []))
        has_current_none = "current_a" in tool_text and "'None'" in tool_text
        print("--- 验证 ---")
        print(f"第二轮 hint_step={state['hint_step'] if state else None}")
        print(f"第二轮回答非空: {bool(second_answer.strip())}")
        print(f"第二轮承接原题: {'原题' in round_two['agent_result'].get('_question', '')}")
        print(f"出现 current_a='None': {has_current_none}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
