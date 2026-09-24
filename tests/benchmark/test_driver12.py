from __future__ import annotations

from types import SimpleNamespace

import pytest

from bg6022.agent import AgentResponse
from bg6022.benchmark.dataset12 import ScenarioTurn
from bg6022.benchmark.e2e12 import (
    Benchmark12DriverError,
    PublicConversationDriver,
)
from bg6022.tools.registry import build_registry


class FakeLlm:
    def __init__(self) -> None:
        self.calls = []


class FakeAgent:
    def __init__(self) -> None:
        self.messages = []
        self.run = _run("waiting", "confirmation")

    def handle_message(self, message: str) -> AgentResponse:
        self.messages.append(message)
        if message == "确认":
            self.run = _run("succeeded", None)
            return AgentResponse("Finished.", run=self.run)
        if message == "迭代上限设为 100":
            return AgentResponse("请明确几何还是 SCF 迭代；当前任务未更改。", run=self.run)
        return AgentResponse("Waiting for confirmation.", run=self.run)


def _run(status: str, waiting_for: str | None):
    plan = SimpleNamespace(steps=[], model_dump=lambda **_kwargs: {"id": "same-plan"})
    return SimpleNamespace(
        id="run-1",
        status=status,
        waiting_for=waiting_for,
        attempts=[],
        plan=plan,
    )


def test_driver_confirms_only_through_public_message() -> None:
    agent = FakeAgent()
    driver = PublicConversationDriver(agent, FakeLlm(), build_registry())
    turns = driver.run([ScenarioTurn(message_template="optimize water", confirm_after=True)])
    assert agent.messages == ["optimize water", "确认"]
    assert len(turns) == 2
    assert turns[0].run_status == "waiting"
    assert turns[0].waiting_for == "confirmation"
    assert turns[0].orca_attempts_after_turn == 0
    assert turns[1].run_status == "succeeded"


def test_driver_does_not_auto_confirm_after_unchanged_clarification() -> None:
    agent = FakeAgent()
    driver = PublicConversationDriver(agent, FakeLlm(), build_registry())
    turns = driver.run(
        [
            ScenarioTurn(message_template="optimize water"),
            ScenarioTurn(message_template="迭代上限设为 100", confirm_after=True),
        ]
    )
    assert agent.messages == ["optimize water", "迭代上限设为 100"]
    assert len(turns) == 2
    assert turns[-1].waiting_for == "confirmation"


def test_driver_turn_bound_includes_public_confirmation() -> None:
    driver = PublicConversationDriver(FakeAgent(), FakeLlm(), build_registry(), max_turns=1)
    with pytest.raises(Benchmark12DriverError, match="turn bound"):
        driver.run([ScenarioTurn(message_template="optimize water", confirm_after=True)])


def test_driver_honors_script_stop_after() -> None:
    agent = FakeAgent()
    driver = PublicConversationDriver(agent, FakeLlm(), build_registry())
    driver.run(
        [
            ScenarioTurn(message_template="first", stop_after=True),
            ScenarioTurn(message_template="must not run", confirm_after=True),
        ]
    )
    assert agent.messages == ["first"]
