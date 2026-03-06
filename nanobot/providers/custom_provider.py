"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import uuid
from typing import Any, Callable

import json_repair
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        # Keep affinity stable for this provider instance to improve backend cache locality.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None, on_token: Callable[[str], None] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        # If on_token callback provided, use streaming mode
        if on_token:
            kwargs["stream"] = True
            try:
                return await self._chat_streaming(**kwargs, on_token=on_token)
            except Exception as e:
                return LLMResponse(content=f"Error: {e}", finish_reason="error")
        else:
            # Non-streaming fallback
            try:
                return self._parse(await self._client.chat.completions.create(**kwargs))
            except Exception as e:
                return LLMResponse(content=f"Error: {e}", finish_reason="error")

    async def _chat_streaming(self, on_token: Callable[[str], None], **kwargs) -> LLMResponse:
        """Stream chat completion and collect full response."""
        full_content = ""
        tool_calls_data: list[dict] = []
        finish_reason = "stop"
        usage = {}
        reasoning_content = None

        stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Handle content delta
            if delta.content:
                full_content += delta.content
                on_token(delta.content)

            # Handle reasoning content (for models that support it)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if reasoning_content is None:
                    reasoning_content = ""
                reasoning_content += delta.reasoning_content

            # Handle tool calls delta
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    tc_index = tc_delta.index
                    # Ensure we have enough slots
                    while len(tool_calls_data) <= tc_index:
                        tool_calls_data.append({"id": "", "function": {"name": "", "arguments": ""}})

                    if tc_delta.id:
                        tool_calls_data[tc_index]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data[tc_index]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_data[tc_index]["function"]["arguments"] += tc_delta.function.arguments

            # Track finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # Track usage (usually only in last chunk)
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens
                }

        # Parse tool calls
        tool_calls = []
        for tc in tool_calls_data:
            if tc.get("id") and tc["function"].get("name"):
                args_str = tc["function"].get("arguments", "{}")
                try:
                    args = json_repair.loads(args_str) if isinstance(args_str, str) else args_str
                except:
                    args = {}
                tool_calls.append(ToolCallRequest(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args
                ))

        return LLMResponse(
            content=full_content if full_content else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model
