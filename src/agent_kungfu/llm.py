from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

# This project is intentionally local-only; avoid LiteLLM's startup fetch of a
# remote pricing table before importing the package.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from litellm import completion  # noqa: E402

from .config import Settings


@dataclass(slots=True)
class OllamaLLM:
    settings: Settings
    timeout: float = 180.0

    def complete(
        self,
        user: str,
        *,
        system: str = "You are a precise data agent.",
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = completion(
            model=self.settings.litellm_model,
            custom_llm_provider="openai",
            base_url=self.settings.ollama_base_url,
            api_key=self.settings.ollama_api_key,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            reasoning_effort="low",
            allowed_openai_params=["reasoning_effort"],
            max_tokens=max_tokens,
            timeout=self.timeout,
            **kwargs,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return str(content).strip()

    async def acomplete(self, *args: Any, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.complete, *args, **kwargs)


def extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    start = re.search(r"\b(select|with)\b", candidate, re.IGNORECASE)
    if not start:
        raise ValueError("No SELECT/CTE SQL found in model output")
    return candidate[start.start() :].strip().rstrip(";")


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
