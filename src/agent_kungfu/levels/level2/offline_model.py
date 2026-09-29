"""A scripted model boundary; the real Agents SDK still executes every tool."""
import json
from uuid import uuid4

from agents import Model, ModelResponse
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText


class LessonModel(Model):
    def __init__(self, n: int):
        self.n = n

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, **kwargs):
        if tools:
            tool = tools[0]  # SDK filters enabled tools using the actual workspace state.
            properties = tool.params_json_schema.get("properties", {})
            arguments = {"n": self.n} if "n" in properties else (
                {"input": f"Return top {self.n} categories"} if "input" in properties else {}
            )
            output = [ResponseFunctionToolCall(
                type="function_call", name=tool.name, arguments=json.dumps(arguments),
                call_id=str(uuid4()),
            )]
        else:
            evidence = next((item.get("output", "") for item in reversed(input)
                             if isinstance(item, dict) and item.get("type") == "function_call_output"), "")
            output = [ResponseOutputMessage(
                id=str(uuid4()), type="message", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text=str(evidence), annotations=[])],
            )]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("The reproducible lesson uses non-streaming Runner.run")
        yield  # Async generator required by Model's streaming interface.
