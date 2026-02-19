from __future__ import annotations

import argparse
import asyncio

from agents import Runner

from agent_trade_sdk.agent import build_trading_agent


async def run_once(prompt: str, model: str | None = None) -> str:
    agent = build_trading_agent(model_name=model)
    result = await Runner.run(agent, prompt)
    return str(result.final_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prompt against the trading agent.")
    parser.add_argument("--prompt", required=True, help="User prompt for the agent run.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    args = parser.parse_args()

    output = asyncio.run(run_once(args.prompt, args.model))
    print(output)


if __name__ == "__main__":
    main()
