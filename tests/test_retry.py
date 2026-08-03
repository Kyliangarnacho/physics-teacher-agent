"""独立有限重试工具的纯本地测试。"""

from __future__ import annotations

import unittest

from src.retry import RetryOutcome, run_with_one_retry


class RetryTests(unittest.TestCase):
    def test_first_attempt_success(self) -> None:
        outcome = run_with_one_retry(lambda: "ok", lambda error: True)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.attempts, 1)
        self.assertEqual(outcome.value, "ok")
        self.assertIsNone(outcome.error)

    def test_retryable_first_failure_then_success(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary")
            return "recovered"

        outcome = run_with_one_retry(operation, lambda error: True)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(outcome.value, "recovered")
        self.assertIsNone(outcome.error)

    def test_both_attempts_fail(self) -> None:
        errors = [TimeoutError("first"), RuntimeError("second")]

        def operation() -> None:
            raise errors.pop(0)

        outcome = run_with_one_retry(operation, lambda error: True)

        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.attempts, 2)
        self.assertIsNone(outcome.value)
        self.assertIsInstance(outcome.error, RuntimeError)
        self.assertEqual(str(outcome.error), "second")

    def test_non_retryable_failure_runs_once(self) -> None:
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1
            raise ValueError("invalid")

        outcome = run_with_one_retry(operation, lambda error: False)

        self.assertEqual(calls, 1)
        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.attempts, 1)

    def test_operation_runs_at_most_twice(self) -> None:
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError("still failing")

        run_with_one_retry(operation, lambda error: True)

        self.assertEqual(calls, 2)

    def test_should_retry_receives_first_exception(self) -> None:
        first_error = TimeoutError("temporary")
        received: list[Exception] = []

        def operation() -> None:
            raise first_error

        def should_retry(error: Exception) -> bool:
            received.append(error)
            return False

        run_with_one_retry(operation, should_retry)

        self.assertEqual(received, [first_error])

    def test_positive_delay_sleeps_once_only_when_retrying(self) -> None:
        calls = 0
        delays: list[float] = []

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary")
            return "ok"

        run_with_one_retry(
            operation,
            lambda error: True,
            delay_seconds=0.25,
            sleep_func=delays.append,
        )

        self.assertEqual(delays, [0.25])

    def test_zero_delay_does_not_call_sleep(self) -> None:
        calls = 0
        delays: list[float] = []

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary")
            return "ok"

        run_with_one_retry(
            operation,
            lambda error: True,
            sleep_func=delays.append,
        )

        self.assertEqual(delays, [])

    def test_negative_delay_is_rejected_before_operation(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            return "unexpected"

        with self.assertRaisesRegex(ValueError, "不得为负数"):
            run_with_one_retry(operation, lambda error: True, delay_seconds=-0.1)

        self.assertEqual(calls, 0)

    def test_none_return_value_is_still_success(self) -> None:
        outcome = run_with_one_retry(lambda: None, lambda error: True)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.attempts, 1)
        self.assertIsNone(outcome.value)
        self.assertIsNone(outcome.error)

    def test_final_error_keeps_real_exception_object(self) -> None:
        final_error = RuntimeError("final")
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("first")
            raise final_error

        outcome = run_with_one_retry(operation, lambda error: True)

        self.assertIs(outcome.error, final_error)

    def test_retry_outcome_rejects_invalid_field_combinations(self) -> None:
        error = ValueError("failed")
        invalid_arguments = (
            {"succeeded": True, "attempts": 1, "value": "ok", "error": error},
            {"succeeded": False, "attempts": 1, "value": None, "error": None},
            {"succeeded": False, "attempts": 1, "value": "bad", "error": error},
            {"succeeded": True, "attempts": 0, "value": "ok", "error": None},
            {"succeeded": True, "attempts": 3, "value": "ok", "error": None},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    RetryOutcome(**arguments)


if __name__ == "__main__":
    unittest.main()
