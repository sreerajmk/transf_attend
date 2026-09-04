"""Bounded LangGraph engineer loop for a software-simulated TCP/IP data path.

The LLM proposes engineering actions, but packet processing is deterministic and
validated locally. Set OPENAI_API_KEY before running the engineer loop.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import os
import random
from pathlib import Path
from collections import deque
from enum import StrEnum
from typing import Any, Callable, Deque, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"level": record.levelname, "event": record.getMessage(), "logger": record.name})


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, state: "AgentState") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(state.model_dump_json(), encoding="utf-8")
        temporary.replace(self.path)

    def load(self) -> "AgentState":
        return AgentState.model_validate_json(self.path.read_text(encoding="utf-8"))


class ToolName(StrEnum):
    READ_REPOSITORY = "read_repository"
    WRITE_FILE = "write_file"
    RUN_TESTS = "run_tests"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = Field(default=3, ge=1, le=10)
    base_delay_seconds: float = Field(default=0.25, ge=0, le=60)
    max_delay_seconds: float = Field(default=8, ge=0, le=120)


class Protocol(StrEnum):
    TCP = "TCP"


class TcpSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_port: int = Field(ge=1, le=65535)
    destination_port: int = Field(ge=1, le=65535)
    sequence: int = Field(ge=0)
    acknowledgement: int = Field(default=0, ge=0)
    flags: frozenset[str] = frozenset({"ACK"})
    payload: bytes = b""

    @field_validator("flags")
    @classmethod
    def valid_flags(cls, value: frozenset[str]) -> frozenset[str]:
        allowed = {"SYN", "ACK", "FIN", "RST"}
        if not value or not value.issubset(allowed):
            raise ValueError("flags must contain only SYN, ACK, FIN, or RST")
        return value


class IpPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: IPvAnyAddress
    destination: IPvAnyAddress
    protocol: Protocol = Protocol.TCP
    ttl: int = Field(default=64, ge=1, le=255)
    segment: TcpSegment


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_mac: str
    destination_mac: str
    payload: IpPacket
    checksum: str

    @field_validator("source_mac", "destination_mac")
    @classmethod
    def valid_mac(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 6 or any(len(part) != 2 for part in parts):
            raise ValueError("MAC address must contain six hex octets")
        int("".join(parts), 16)
        return value.lower()


class SimulatedBuffer:
    """Bounded FIFO buffer; it models the link without using real sockets."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._frames: Deque[Frame] = deque(maxlen=capacity)
        self.capacity = capacity
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        if len(self._frames) >= self.capacity:
            self.dropped += 1
            raise BufferError("simulated link buffer is full")
        self._frames.append(frame)

    def get(self) -> Frame:
        if not self._frames:
            raise BufferError("simulated link buffer is empty")
        return self._frames.popleft()

    def __len__(self) -> int:
        return len(self._frames)


class TcpIpStack:
    """Small deterministic IPv4/TCP-like stack for exercising data flow."""

    def __init__(self, address: str, mac: str, link: SimulatedBuffer) -> None:
        self.address = ipaddress.ip_address(address)
        self.mac = mac
        self.link = link
        self._next_sequence = 1

    def send(self, destination: str, destination_mac: str, port: int, data: bytes) -> None:
        segment = TcpSegment(
            source_port=49152,
            destination_port=port,
            sequence=self._next_sequence,
            flags=frozenset({"ACK"}),
            payload=data,
        )
        self._next_sequence += len(data)
        packet = IpPacket(source=self.address, destination=destination, segment=segment)
        checksum = hashlib.sha256(packet.model_dump_json().encode()).hexdigest()
        self.link.put(Frame(source_mac=self.mac, destination_mac=destination_mac, payload=packet, checksum=checksum))

    def receive(self) -> bytes:
        frame = self.link.get()
        expected = hashlib.sha256(frame.payload.model_dump_json().encode()).hexdigest()
        if frame.checksum != expected:
            raise ValueError("frame checksum validation failed")
        if frame.payload.ttl <= 1:
            raise ValueError("packet TTL expired")
        return frame.payload.segment.payload


class Action(StrEnum):
    INSPECT = "inspect"
    IMPLEMENT = "implement"
    TEST = "test"
    FINISH = "finish"


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    rationale: str = Field(min_length=1, max_length=1000)
    progress: str = Field(min_length=1, max_length=1000)
    requested_tools: frozenset[ToolName] = frozenset()
    done: bool = False


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1)
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=8, ge=1, le=100)
    phase: str = "inspect"
    completed: bool = False
    history: list[AgentDecision] = Field(default_factory=list)
    last_error: str | None = None


class GraphState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    phase: str
    completed: bool
    history: list[dict[str, Any]]
    last_error: str | None


def _validated_state(raw: GraphState) -> AgentState:
    return AgentState.model_validate(raw)


def build_engineer_graph(
    model: ChatOpenAI,
    *,
    allowed_tools: frozenset[ToolName] = frozenset(),
    checkpoint: CheckpointStore | None = None,
    retry_policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = __import__("time").sleep,
):
    """Build a bounded self-prompting graph. The graph never trusts raw model JSON."""

    structured_model = model.with_structured_output(AgentDecision)

    def engineer_node(raw: GraphState) -> GraphState:
        state = _validated_state(raw)
        if state.completed or state.iteration >= state.max_iterations:
            return state.model_dump()
        prompt = (
            "You are the implementation engineer. Continue this task one bounded step at a time.\n"
            f"Task: {state.task}\nPhase: {state.phase}\nIteration: {state.iteration + 1}/{state.max_iterations}\n"
            "Choose inspect, implement, test, or finish. Never claim done before test evidence.\n"
            f"You may request only these tools: {sorted(tool.value for tool in allowed_tools)}\n"
            f"Previous progress: {[item.progress for item in state.history[-3:]]}"
        )
        policy = retry_policy or RetryPolicy()
        try:
            response = None
            for attempt in range(policy.attempts):
                try:
                    response = structured_model.invoke(prompt)
                    break
                except Exception:
                    LOGGER.warning("model_request_failed attempt=%d", attempt + 1)
                    if attempt + 1 == policy.attempts:
                        raise
                    delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2**attempt))
                    sleep(delay * random.uniform(0.8, 1.2))
            decision = AgentDecision.model_validate(response)
            unauthorized = decision.requested_tools - allowed_tools
            if unauthorized:
                raise ValueError(f"unauthorized tools requested: {sorted(tool.value for tool in unauthorized)}")
            history = [*state.history, decision]
            # Repeating the same non-terminal action means the model is not progressing.
            repeated = len(history) >= 2 and history[-1].model_dump() == history[-2].model_dump()
            completed = decision.done or decision.action is Action.FINISH or repeated
            new_state = AgentState(
                **state.model_dump(exclude={"history", "iteration", "phase", "completed", "last_error"}),
                iteration=state.iteration + 1,
                phase=decision.action.value,
                completed=completed,
                history=history,
                last_error="repeated decision detected" if repeated else None,
            )
            if checkpoint:
                checkpoint.save(new_state)
            LOGGER.info("engineer_step iteration=%d action=%s", new_state.iteration, decision.action.value)
            return new_state.model_dump()
        except (ValidationError, TypeError, ValueError) as exc:
            LOGGER.warning("model_decision_rejected reason=%s", str(exc))
            new_state = AgentState(
                **state.model_dump(exclude={"history", "iteration", "phase", "completed", "last_error"}),
                iteration=state.iteration + 1,
                phase=Action.TEST.value,
                completed=state.iteration + 1 >= state.max_iterations,
                history=state.history,
                last_error=f"invalid model decision: {exc}",
            )
            if checkpoint:
                checkpoint.save(new_state)
            return new_state.model_dump()

    def route(raw: GraphState) -> str:
        state = _validated_state(raw)
        return END if state.completed or state.iteration >= state.max_iterations else "engineer"

    graph = StateGraph(GraphState)
    graph.add_node("engineer", engineer_node)
    graph.add_edge(START, "engineer")
    graph.add_conditional_edges("engineer", route)
    return graph.compile()


def run_engineer_loop(
    task: str,
    max_iterations: int = 8,
    *,
    checkpoint_path: str | Path | None = "agent-checkpoint.json",
) -> AgentState:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the engineer loop")
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        max_retries=0,
    )
    checkpoint = CheckpointStore(checkpoint_path) if checkpoint_path else None
    initial = checkpoint.load() if checkpoint and checkpoint.path.exists() else AgentState(task=task, max_iterations=max_iterations)
    result = build_engineer_graph(
        model,
        allowed_tools=frozenset(),
        checkpoint=checkpoint,
        retry_policy=RetryPolicy(),
    ).invoke(initial.model_dump())
    return AgentState.model_validate(result)


def run_data_flow(data: bytes) -> bytes:
    link = SimulatedBuffer(capacity=4)
    sender = TcpIpStack("10.0.0.1", "02:00:00:00:00:01", link)
    receiver = TcpIpStack("10.0.0.2", "02:00:00:00:00:02", link)
    sender.send("10.0.0.2", receiver.mac, 8080, data)
    return receiver.receive()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="hello over simulated TCP/IP")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--skip-agent", action="store_true", help="run only the deterministic data-flow demo")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    for handler in logging.getLogger().handlers:
        handler.setFormatter(JsonFormatter())
    print(run_data_flow(args.data.encode()).decode())
    if not args.skip_agent:
        state = run_engineer_loop("Implement and verify the simulated TCP/IP data path", args.max_iterations)
        print(state.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
