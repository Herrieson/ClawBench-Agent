from __future__ import annotations

import argparse

from agent_system import build_default_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple agent harness for ClawBench.")
    parser.add_argument(
        "message",
        nargs="*",
        help="Optional one-shot user message. If empty, start interactive mode.",
    )
    args = parser.parse_args()

    agent = build_default_agent()

    if args.message:
        user_input = " ".join(args.message)
        print(agent.run(user_input))
        return

    print("SimpleAgent REPL. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("bye")
            break

        print(f"agent> {agent.run(user_input)}")


if __name__ == "__main__":
    main()
