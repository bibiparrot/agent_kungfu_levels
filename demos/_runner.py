from __future__ import annotations

import argparse
import json

from agent_kungfu.levels import DEFAULT_QUESTION, run_level


def main(level: int) -> None:
    parser = argparse.ArgumentParser(description=f"Run agent kung-fu level {level}")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--deepeval", action="store_true")
    args = parser.parse_args()
    result = run_level(level, args.question, offline=args.offline, use_deepeval=args.deepeval)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))

