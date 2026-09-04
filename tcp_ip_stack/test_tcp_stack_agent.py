from pathlib import Path

import pytest

from tcp_stack_agent import (
    Action,
    AgentDecision,
    AgentState,
    CheckpointStore,
    Frame,
    IpPacket,
    RetryPolicy,
    SimulatedBuffer,
    TcpIpStack,
    ToolName,
    build_engineer_graph,
    run_data_flow,
)


class FakeStructuredModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        self.calls += 1
        return next(self.responses)


def decision(**overrides):
    value = {
        "action": "test",
        "rationale": "run the tests",
        "progress": "test step completed",
        "done": False,
    }
    value.update(overrides)
    return value


def test_data_flow_round_trip():
    assert run_data_flow(b"payload") == b"payload"


def test_buffer_overflow_is_explicit():
    buffer = SimulatedBuffer(capacity=1)
    stack = TcpIpStack("10.0.0.1", "02:00:00:00:00:01", buffer)
    stack.send("10.0.0.2", "02:00:00:00:00:02", 80, b"one")
    with pytest.raises(BufferError, match="full"):
        stack.send("10.0.0.2", "02:00:00:00:00:02", 80, b"two")
    assert buffer.dropped == 1


def test_checksum_failure_is_rejected():
    buffer = SimulatedBuffer()
    sender = TcpIpStack("10.0.0.1", "02:00:00:00:00:01", buffer)
    sender.send("10.0.0.2", "02:00:00:00:00:02", 80, b"data")
    frame = buffer.get()
    tampered = Frame(**{**frame.model_dump(), "checksum": "bad"})
    buffer.put(tampered)
    receiver = TcpIpStack("10.0.0.2", "02:00:00:00:00:02", buffer)
    with pytest.raises(ValueError, match="checksum"):
        receiver.receive()


def test_malformed_model_output_is_contained():
    model = FakeStructuredModel([{"unexpected": "shape"}])
    state = build_engineer_graph(model).invoke({"task": "test", "max_iterations": 1})
    validated = AgentState.model_validate(state)
    assert validated.completed is True
    assert validated.last_error and "invalid model decision" in validated.last_error


def test_repeated_decisions_terminate():
    same = decision()
    model = FakeStructuredModel([same, same])
    state = build_engineer_graph(model).invoke({"task": "test", "max_iterations": 10})
    validated = AgentState.model_validate(state)
    assert validated.completed is True
    assert validated.iteration == 2
    assert validated.last_error == "repeated decision detected"


def test_hard_iteration_limit_terminates():
    model = FakeStructuredModel([decision(progress=f"step {index}") for index in range(3)])
    state = build_engineer_graph(model).invoke({"task": "test", "max_iterations": 3})
    assert AgentState.model_validate(state).iteration == 3
    assert model.calls == 3


def test_unauthorized_tool_is_rejected():
    model = FakeStructuredModel([decision(requested_tools=[ToolName.WRITE_FILE])])
    state = build_engineer_graph(model, allowed_tools=frozenset()).invoke({"task": "test", "max_iterations": 1})
    assert "unauthorized tools" in AgentState.model_validate(state).last_error


def test_checkpoint_is_written_and_loadable(tmp_path: Path):
    checkpoint = CheckpointStore(tmp_path / "state.json")
    model = FakeStructuredModel([decision(done=True, action=Action.FINISH)])
    build_engineer_graph(model, checkpoint=checkpoint).invoke({"task": "test", "max_iterations": 2})
    loaded = checkpoint.load()
    assert loaded.completed is True
    assert loaded.history[0].action is Action.FINISH


def test_retry_backoff_retries_transient_model_failure():
    class FlakyModel(FakeStructuredModel):
        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary")
            return decision(done=True, action=Action.FINISH)

    delays = []
    model = FlakyModel([])
    state = build_engineer_graph(
        model,
        retry_policy=RetryPolicy(attempts=2, base_delay_seconds=1, max_delay_seconds=1),
        sleep=delays.append,
    ).invoke({"task": "test", "max_iterations": 1})
    assert AgentState.model_validate(state).completed is True
    assert model.calls == 2
    assert len(delays) == 1
