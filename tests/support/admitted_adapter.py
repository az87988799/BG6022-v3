from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from bg6022 import execution
from bg6022.agent import _step_fingerprint
from bg6022.config import AppConfig
from bg6022.models import Result, Run, Step, Tool

T = TypeVar("T", bound=Result)


def execute_adapter_under_test_gateway(
    config: AppConfig,
    run: Run,
    step: Step,
    tool: Tool,
    execute: Callable[[], T],
) -> T:
    """Give a focused adapter test the same owner context as Agent.advance."""

    run.status = "running"
    owner = execution.RunOwner(config.data_root_path, run.id)
    with owner:
        if not tool.reserve_attempt(run, step):
            raise AssertionError("test adapter entry was not admitted by its Tool budget")
        context = execution.begin_attempt(
            config.data_root_path,
            run,
            step,
            owner=owner,
            budget_reserved=True,
            step_fingerprint=_step_fingerprint(step),
        )
        try:
            result = execute()
            result.step_fingerprint = _step_fingerprint(step)
            result.input_bindings = {
                name: reference.artifact_id
                for name, reference in step.inputs.items()
                if reference.artifact_id is not None
            }
            result.input_artifact_ids = list(result.input_bindings.values())
            execution.finish_attempt(run, context, result, persist=False, release=False)
            execution.commit_attempt_result(
                config.data_root_path,
                run,
                step,
                result,
                expected_input_bindings=result.input_bindings,
                tool=tool,
                context=context,
            )
            return result
        except Exception as error:
            if execution.active_attempt(run, step.id) is context:
                execution.fail_attempt(
                    run,
                    context,
                    category="test_adapter_failure",
                    reason=str(error),
                    persist=True,
                )
            raise
        finally:
            if execution.active_attempt(run, step.id) is context:
                execution.release_attempt(run, context)
