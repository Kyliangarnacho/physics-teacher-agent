"""Stage 05 本地 BM25 知识检索器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge" / "physics_notes_v1.jsonl"
SEARCH_FIELDS = ("chapter", "topic", "content")
STOPWORDS = {
    "一个",
    "一条",
    "为什么",
    "了",
    "在",
    "如何",
    "怎么",
    "怎样",
    "把",
    "是",
    "时",
    "哪里",
    "中",
    "与",
    "和",
    "或",
    "的",
    "被",
}


def tokenize(text: str) -> list[str]:
    """使用 jieba 分词，并移除空白、标点和少量通用停用词。"""
    tokens: list[str] = []
    for raw_token in jieba.lcut(text):
        token = raw_token.strip()
        if (
            token
            and token not in STOPWORDS
            and any(character.isalnum() for character in token)
        ):
            tokens.append(token)
    return tokens


def load_knowledge_base(
    path: str | Path = DEFAULT_KNOWLEDGE_BASE,
) -> list[dict[str, Any]]:
    """从项目内的 JSONL 文件加载知识卡片。"""
    knowledge_path = Path(path)
    if not knowledge_path.is_absolute():
        knowledge_path = PROJECT_ROOT / knowledge_path

    cards: list[dict[str, Any]] = []
    with knowledge_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                card = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"知识库第 {line_number} 行不是合法 JSON。"
                ) from error
            if not isinstance(card, dict):
                raise ValueError(f"知识库第 {line_number} 行必须是 JSON 对象。")
            cards.append(card)

    return cards


class KnowledgeRetriever:
    """加载知识卡片并提供 BM25 排序检索。"""

    def __init__(self, path: str | Path = DEFAULT_KNOWLEDGE_BASE) -> None:
        self.cards = load_knowledge_base(path)
        if not self.cards:
            raise ValueError("知识库为空，无法建立检索索引。")

        corpus = [tokenize(self._build_search_text(card)) for card in self.cards]
        self.index = BM25Okapi(corpus)

    @staticmethod
    def _build_search_text(card: dict[str, Any]) -> str:
        """合并章节、主题、关键词和正文作为检索文本。"""
        keywords = card.get("keywords", [])
        keyword_text = " ".join(str(keyword) for keyword in keywords)
        text_parts = [str(card.get(field, "")) for field in SEARCH_FIELDS]
        text_parts.insert(2, keyword_text)
        return " ".join(text_parts)

    def search(
        self,
        question: str,
        top_k: int = 3,
        min_score_ratio: float = 0.3,
    ) -> list[dict[str, Any]]:
        """返回达到最高分相对阈值的 BM25 知识卡片。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("检索问题不能为空。")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k 必须是正整数。")
        if top_k > len(self.cards):
            raise ValueError(
                f"top_k 不能超过知识卡片数量 {len(self.cards)}。"
            )
        if (
            isinstance(min_score_ratio, bool)
            or not isinstance(min_score_ratio, (int, float))
            or not 0 <= min_score_ratio <= 1
        ):
            raise ValueError("min_score_ratio 必须是 0 到 1 之间的数字。")

        scores = self.index.get_scores(tokenize(question))
        positive_ranked = sorted(
            (
                (card_index, float(score))
                for card_index, score in enumerate(scores)
                if float(score) > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not positive_ranked:
            return []

        highest_score = positive_ranked[0][1]
        minimum_score = highest_score * min_score_ratio
        filtered_ranked = [
            item for item in positive_ranked if item[1] >= minimum_score
        ][:top_k]

        results = []
        for card_index, score in filtered_ranked:
            result = dict(self.cards[card_index])
            result["score"] = score
            results.append(result)

        return results
