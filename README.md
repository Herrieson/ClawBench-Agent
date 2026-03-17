# ClawBench-Agent

A minimal Python agent harness for testing simple ClawBench-style agent flows.

## Quick Start

```bash
uv run python main.py
```

or run a one-shot message:

```bash
uv run python main.py "calc 1 + 2 * 3"
```

## Built-in Commands

- `calc 1 + 2 * 3`
- `time`
- `echo hello`
- `/tool <tool_name> <tool_input>`

## Files

- `main.py`: CLI entrypoint and REPL.
- `agent_system.py`: simple agent loop, rule-based model, and tool registry.
