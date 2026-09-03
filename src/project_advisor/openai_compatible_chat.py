"""Pure-Python OpenAI-compatible chat model that does not import openai/jiter.

Some managed Windows environments block unsigned native extensions such as
``jiter.pyd``. DeepSeek and OpenAI chat-completions are HTTP APIs, so the
project can use HTTPX directly while retaining LangChain's BaseChatModel,
tool-calling and structured-output contracts.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

from project_advisor import __version__


def _message_content(content: Any) -> Any:
    if isinstance(content, (str, list)):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _message_to_payload(message: BaseMessage) -> dict[str, Any]:
    if message.type == "human":
        role = "user"
    elif message.type == "system":
        role = "system"
    elif message.type == "tool":
        role = "tool"
    else:
        role = "assistant"

    payload: dict[str, Any] = {
        "role": role,
        "content": _message_content(message.content),
    }
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("args", {}), ensure_ascii=False, default=str
                    ),
                },
            }
            for index, call in enumerate(message.tool_calls)
        ]
    return payload


def _tool_choice_payload(tool_choice: Any) -> Any:
    if tool_choice in (None, False):
        return None
    if tool_choice is True or tool_choice == "any":
        return "required"
    if isinstance(tool_choice, str) and tool_choice not in {
        "auto",
        "none",
        "required",
    }:
        return {"type": "function", "function": {"name": tool_choice}}
    return tool_choice


class OpenAICompatibleChatModel(BaseChatModel):
    """Minimal chat-completions client with LangChain tool-call support."""

    model_name: str = Field(alias="model")
    api_key: SecretStr = Field(repr=False)
    base_url: str
    max_tokens: int = 4096
    timeout_seconds: float = 60.0
    default_extra_body: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "project-advisor-openai-compatible"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model_name, "base_url": self.base_url}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ):
        kwargs.pop("ls_structured_output_format", None)
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        normalized_choice = _tool_choice_payload(tool_choice)
        if normalized_choice is not None:
            kwargs["tool_choice"] = normalized_choice
        return self.bind(tools=formatted_tools, **kwargs)

    def _request_payload(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [_message_to_payload(message) for message in messages],
            "max_tokens": self.max_tokens,
            **self.default_extra_body,
        }
        for key in (
            "tools",
            "tool_choice",
            "temperature",
            "top_p",
            "response_format",
            "frequency_penalty",
            "presence_penalty",
        ):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]
        if stop:
            payload["stop"] = stop
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"project-advisor/{__version__}",
        }

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _result(data: dict[str, Any]) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("模型响应不包含 choices。")
        response_message = choices[0].get("message") or {}
        tool_calls = []
        invalid_tool_calls = []
        for raw_call in response_message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象。")
                tool_calls.append({
                    "name": function.get("name", ""),
                    "args": arguments,
                    "id": raw_call.get("id"),
                    "type": "tool_call",
                })
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                invalid_tool_calls.append({
                    "name": function.get("name"),
                    "args": function.get("arguments"),
                    "id": raw_call.get("id"),
                    "error": str(error),
                    "type": "invalid_tool_call",
                })

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        message = AIMessage(
            content=response_message.get("content") or "",
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={
                "model_name": data.get("model"),
                "finish_reason": choices[0].get("finish_reason"),
                "token_usage": usage,
            },
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(
                    usage.get("total_tokens", input_tokens + output_tokens) or 0
                ),
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"model_name": data.get("model"), "token_usage": usage},
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = response.text[:1000]
            raise RuntimeError(
                f"模型 API 请求失败（HTTP {response.status_code}）：{detail}"
            ) from error

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                self._endpoint(),
                headers=self._headers(),
                json=self._request_payload(messages, stop, kwargs),
            )
        self._raise_for_status(response)
        return self._result(response.json())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self._endpoint(),
                headers=self._headers(),
                json=self._request_payload(messages, stop, kwargs),
            )
        self._raise_for_status(response)
        return self._result(response.json())
