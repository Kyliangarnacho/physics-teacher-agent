"""教师事实约束与关键知识卡的最小回归测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.prompts import JUNIOR_PHYSICS_SYSTEM_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge" / "physics_notes_v1.jsonl"


class FactualGuardrailTests(unittest.TestCase):
    def test_teacher_prompt_checks_absolute_and_changing_conditions(self) -> None:
        self.assertIn(
            "任何时候、一定、永远、绝不",
            JUNIOR_PHYSICS_SYSTEM_PROMPT,
        )
        self.assertIn("前提错误时先勘误", JUNIOR_PHYSICS_SYSTEM_PROMPT)
        self.assertIn(
            "把明显随温度等条件变化的物理量说成始终不变",
            JUNIOR_PHYSICS_SYSTEM_PROMPT,
        )

    def test_elec_003_keeps_rated_power_and_filament_facts(self) -> None:
        cards = [
            json.loads(line)
            for line in KNOWLEDGE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        card = next(card for card in cards if card["id"] == "KB-ELEC-003")
        content = card["content"]

        self.assertIn(
            "额定功率是用电器在额定电压下正常工作时的功率",
            content,
        )
        self.assertIn("实际功率等于额定功率", content)
        self.assertIn("其他工作条件下实际功率可能不同", content)
        self.assertIn("白炽灯丝电阻会随温度变化", content)
        self.assertIn("不能简单视为始终不变", content)


if __name__ == "__main__":
    unittest.main()
