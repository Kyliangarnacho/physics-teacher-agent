"""独立、有限且不包含业务异常策略的重试工具。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryOutcome(Generic[T]):
    """保存一次操作在最多两次尝试后的结果。"""

    succeeded: bool
    attempts: int
    value: T | None
    error: Exception | None

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise ValueError("succeeded 必须是布尔值。")
        if isinstance(self.attempts, bool) or self.attempts not in {1, 2}:
            raise ValueError("attempts 只能是 1 或 2。")
        if self.succeeded:
            if self.error is not None:
                raise ValueError("成功结果的 error 必须为 None。")
            return
        if self.value is not None:
            raise ValueError("失败结果的 value 必须为 None。")
        if not isinstance(self.error, Exception):
            raise ValueError("失败结果必须保留真实异常对象。")


def run_with_one_retry(
    operation: Callable[[], T],
    should_retry: Callable[[Exception], bool],
    delay_seconds: float = 0.0,
    sleep_func: Callable[[float], object] = time.sleep,
) -> RetryOutcome[T]:
    """执行操作，并在策略允许时额外尝试一次。"""
    if delay_seconds < 0:
        raise ValueError("delay_seconds 不得为负数。")

    try:
        value = operation()
    except Exception as first_error:
        if not should_retry(first_error):
            return RetryOutcome(
                succeeded=False,
                attempts=1,
                value=None,
                error=first_error,
            )

        if delay_seconds > 0:
            sleep_func(delay_seconds)

        try:
            value = operation()
        except Exception as final_error:
            return RetryOutcome(
                succeeded=False,
                attempts=2,
                value=None,
                error=final_error,
            )
        return RetryOutcome(
            succeeded=True,
            attempts=2,
            value=value,
            error=None,
        )

    return RetryOutcome(
        succeeded=True,
        attempts=1,
        value=value,
        error=None,
    )
