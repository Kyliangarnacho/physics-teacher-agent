"""Stage 10 记忆页面交互真实探针（千问 API，AppTest 驱动，临时数据库）。

验证四道门：
1. 不点击“分析本轮学习表现”不产生候选；
2. 提取后不点击“确认保存”数据库无新增记忆；
3. 确认后才写入，并能在学习档案中停用；
4. 停用后相关新问题不再使用该记忆。
只打印到 stdout，不写 Git 跟踪文件，不输出 API Key。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from src.memory.retrieval import retrieve_relevant_memories
from src.storage import list_memories


def button_by_key(app: AppTest, key: str):
    return next(item for item in app.button if item.key == key)


def key_with_prefix(app: AppTest, prefix: str) -> str:
    return next(item.key for item in app.button if item.key.startswith(prefix))


def main() -> int:
    print("Stage 10 memory page probe (real API, AppTest, temp db)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ["PHYSICS_AGENT_DB_PATH"] = str(
            Path(tmp_dir) / "physics_teacher.db"
        )
        app = AppTest.from_file("app.py")
        app.run(timeout=180)

        print("=== 门 1：不点击分析不产生候选 ===")
        app.chat_input[0].set_value(
            "为什么物体下落时机械能一定减小？"
        ).run(timeout=180)
        print(
            "analyze buttons:",
            [
                item.key
                for item in app.button
                if item.key.startswith("analyze_memory_")
            ],
        )
        print(
            "candidates before click:",
            dict(
                app.session_state["memory_candidates"]
                if "memory_candidates" in app.session_state
                else {}
            ),
        )
        print("memories before click:", len(list_memories()))

        print("=== 门 2：点击分析后暂存候选但不写库 ===")
        button_by_key(
            app,
            key_with_prefix(app, "analyze_memory_"),
        ).click().run(timeout=180)
        entry = next(iter(app.session_state["memory_candidates"].values()))
        print("candidate count:", len(entry["candidates"]), "status:", entry["status"])
        print("memories after extract:", len(list_memories()))
        for candidate in entry["candidates"][:3]:
            print(
                "  ",
                candidate["memory_type"],
                candidate["topic"],
                candidate["confidence"],
            )

        print("=== 门 3：确认后才写入 ===")
        confirm_keys = [
            item.key
            for item in app.button
            if item.key.startswith("confirm_memory_")
        ]
        if confirm_keys:
            button_by_key(app, confirm_keys[0]).click().run(timeout=180)
            memories = list_memories()
            print(
                "memories after confirm:",
                len(memories),
                [
                    (memory["memory_type"], memory["topic"], memory["evidence_count"])
                    for memory in memories
                ],
            )
        else:
            print("没有候选可确认，跳过门 3/4")

        if list_memories():
            print("=== 门 4：停用后不再召回 ===")
            memory = list_memories()[0]
            button_by_key(
                app,
                f"memory_deactivate_{memory['id']}",
            ).click().run(timeout=180)
            recalled = retrieve_relevant_memories(
                "物体下落机械能怎样变化？"
            )
            print("recalled after deactivate:", len(recalled))
            print(
                "memory active:",
                list_memories(include_inactive=True)[0]["active"],
            )
        print("exceptions:", len(app.exception))
    return 0


if __name__ == "__main__":
    sys.exit(main())
