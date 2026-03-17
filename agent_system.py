from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import ast
import operator
from typing import Callable, Protocol


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


@dataclass
class ModelResponse:
    final_answer: str | None = None
    tool_name: str | None = None
    tool_input: str = ""


class AgentModel(Protocol):
    def generate(self, messages: list[Message], tools: dict[str, Tool]) -> ModelResponse:
        ...


class RuleBasedModel:
    """A deterministic model for baseline agent-flow testing."""

    def generate(self, messages: list[Message], tools: dict[str, Tool]) -> ModelResponse:
        if not messages:
            return ModelResponse(final_answer="No input provided.")

        latest_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if latest_user is None:
            return ModelResponse(final_answer="No user message found.")

        text = latest_user.content.strip()
        lowered = text.lower()

        # Command mode: "/tool <name> <input>"
        if lowered.startswith("/tool "):
            parts = text.split(maxsplit=2)
            if len(parts) < 2:
                return ModelResponse(
                    final_answer="Usage: /tool <tool_name> <tool_input>"
                )
            tool_name = parts[1]
            tool_input = parts[2] if len(parts) > 2 else ""
            return ModelResponse(tool_name=tool_name, tool_input=tool_input)

        # Convenient shortcuts for quick local testing.
        if lowered.startswith("calc "):
            return ModelResponse(tool_name="calc", tool_input=text[5:].strip())
        if lowered in {"time", "now"}:
            return ModelResponse(tool_name="time", tool_input="")
        if lowered.startswith("echo "):
            return ModelResponse(tool_name="echo", tool_input=text[5:].strip())

        tool_list = ", ".join(sorted(tools)) or "none"
        return ModelResponse(
            final_answer=(
                "SimpleAgent received your message.\n"
                f"Available tools: {tool_list}\n"
                "Try: 'calc 1 + 2 * 3', 'time', or '/tool echo hello'."
            )
        )


@dataclass
class Agent:
    model: AgentModel
    tools: dict[str, Tool] = field(default_factory=dict)
    history: list[Message] = field(default_factory=list)

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def run(self, user_input: str, max_steps: int = 3) -> str:
        self.history.append(Message(role="user", content=user_input))

        for _ in range(max_steps):
            decision = self.model.generate(self.history, self.tools)

            if decision.final_answer is not None:
                self.history.append(Message(role="assistant", content=decision.final_answer))
                return decision.final_answer

            if decision.tool_name is None:
                fallback = "Model returned neither final answer nor tool call."
                self.history.append(Message(role="assistant", content=fallback))
                return fallback

            tool = self.tools.get(decision.tool_name)
            if tool is None:
                msg = f"Tool '{decision.tool_name}' not found."
                self.history.append(Message(role="assistant", content=msg))
                return msg

            try:
                tool_output = tool.func(decision.tool_input)
            except Exception as exc:  # noqa: BLE001 - this is a demo harness
                tool_output = f"Tool '{tool.name}' failed: {exc}"

            self.history.append(
                Message(
                    role="tool",
                    content=f"{tool.name}({decision.tool_input!r}) -> {tool_output}",
                )
            )

            # For this simple baseline, we return immediately after one tool call.
            answer = f"[tool:{tool.name}] {tool_output}"
            self.history.append(Message(role="assistant", content=answer))
            return answer

        timeout_msg = f"Stopped after {max_steps} steps without final answer."
        self.history.append(Message(role="assistant", content=timeout_msg))
        return timeout_msg


def _safe_eval_expr(expr: str) -> float:
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operator")
            return op(eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported unary operator")
            return op(eval_node(node.operand))
        raise ValueError("Unsupported expression")

    parsed = ast.parse(expr, mode="eval")
    return eval_node(parsed)


def default_tools() -> list[Tool]:
    return [
        Tool(
            name="echo",
            description="Echo back the tool input text.",
            func=lambda text: text,
        ),
        Tool(
            name="time",
            description="Return current local datetime in ISO format.",
            func=lambda _: datetime.now().isoformat(timespec="seconds"),
        ),
        Tool(
            name="calc",
            description="Evaluate a basic math expression, e.g. 1 + 2 * 3.",
            func=lambda expr: str(_safe_eval_expr(expr)),
        ),
    ]


def build_default_agent() -> Agent:
    agent = Agent(model=RuleBasedModel())
    for tool in default_tools():
        agent.register_tool(tool)
    return agent
