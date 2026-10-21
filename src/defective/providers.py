"""Personal-API LLM providers behind one small interface.

Each provider keeps its own native message history (Anthropic content blocks,
OpenAI chat messages, Gemini Contents) so nothing is lost translating turns;
the agent loop only sees `Turn` objects and sends back tool results.

A provider is a wire format (anthropic | openai | gemini) plus a base URL and
key, so Argo is just configuration: Claude over Argo's /v1/messages, GPT and
Gemini over its OpenAI-compatible /v1. See configs/models.yaml.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = ""          # provider stop reason, for diagnostics


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema object


def load_model_config() -> dict:
    return yaml.safe_load((REPO / "configs/models.yaml").read_text())


class Provider:
    name = ""

    def __init__(self, system: str, tools: list[Tool], cfg: dict):
        self.system, self.tools, self.cfg = system, tools, cfg

    def send_user(self, text: str) -> Turn: ...
    def send_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> Turn: ...


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, system, tools, cfg):
        super().__init__(system, tools, cfg)
        import anthropic
        self.client = anthropic.Anthropic(api_key=cfg["api_key"], base_url=cfg.get("base_url"))
        self.messages: list[dict] = []
        self.api_tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                          for t in tools]

    def _call(self) -> Turn:
        # Streaming because max_tokens is large (and Argo's Claude path requires it).
        kw: dict[str, Any] = {}
        if self.cfg.get("effort"):
            kw.update(thinking={"type": "adaptive"}, output_config={"effort": self.cfg["effort"]})
        if self.cfg.get("fallbacks"):   # Claude API only: re-run refusals on Anthropic's fallback model
            kw.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        with self.client.beta.messages.stream(
            model=self.cfg["model"],
            max_tokens=self.cfg.get("max_tokens", 64000),
            system=self.system,
            tools=self.api_tools,
            messages=self.messages,
            **kw,
        ) as stream:
            msg = stream.get_final_message()
        # Append the full content (thinking blocks included) unchanged: history stays append-only.
        self.messages.append({"role": "assistant", "content": msg.content})
        if msg.stop_reason == "refusal":
            return Turn("[model declined this request]", stop="refusal")
        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in msg.content if b.type == "tool_use"]
        return Turn(text, calls, msg.stop_reason or "")

    def send_user(self, text):
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": out, "is_error": err}
            for c, out, err in results]})
        return self._call()


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, system, tools, cfg):
        super().__init__(system, tools, cfg)
        import openai
        self.client = openai.OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url"))
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.api_tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]

    def _call(self) -> Turn:
        r = self.client.chat.completions.create(
            model=self.cfg["model"], messages=self.messages, tools=self.api_tools,
            max_completion_tokens=self.cfg.get("max_completion_tokens", 16000))
        m = r.choices[0].message
        self.messages.append(m.model_dump(exclude_none=True))
        calls = []
        for tc in m.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return Turn(m.content or "", calls, r.choices[0].finish_reason or "")

    def send_user(self, text):
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results):
        for c, out, _err in results:
            self.messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        return self._call()


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, system, tools, cfg):
        super().__init__(system, tools, cfg)
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=cfg["api_key"])
        self.contents: list = []
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in tools])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _call(self) -> Turn:
        r = self.client.models.generate_content(
            model=self.cfg["model"], contents=self.contents, config=self.config)
        cand = r.candidates[0] if r.candidates else None
        if cand and cand.content:
            self.contents.append(cand.content)   # keeps thought signatures intact
        calls = [ToolCall(fc.id or f"call_{i}", fc.name, dict(fc.args or {}))
                 for i, fc in enumerate(r.function_calls or [])]
        text = "".join(p.text for p in (cand.content.parts if cand and cand.content else [])
                       if getattr(p, "text", None) and not getattr(p, "thought", False))
        return Turn(text, calls, str(cand.finish_reason) if cand else "")

    def send_user(self, text):
        self.contents.append(self.types.Content(role="user", parts=[self.types.Part.from_text(text=text)]))
        return self._call()

    def send_tool_results(self, results):
        parts = [self.types.Part.from_function_response(
            name=c.name, response={"error" if err else "result": out}) for c, out, err in results]
        self.contents.append(self.types.Content(role="user", parts=parts))
        return self._call()


WIRES = {p.name: p for p in (AnthropicProvider, OpenAIProvider, GeminiProvider)}


def resolve(provider: str | None = None, model: str | None = None) -> dict:
    """Merge provider- and model-level config. `model` is an alias or a literal id."""
    cfg = load_model_config()
    name = provider or os.environ.get("DEFECTIVE_PROVIDER") or cfg["default_provider"]
    if name not in cfg["providers"]:
        raise ValueError(f"Unknown provider {name!r}; choose from {sorted(cfg['providers'])}")
    p = cfg["providers"][name]
    model = model or os.environ.get("DEFECTIVE_MODEL") or next(iter(p["models"]))
    m = p["models"].get(model, {"id": model})
    out = {k: v for k, v in p.items() if k != "models"} | m
    out["provider"], out["model"] = name, out.pop("id")
    out["api_key"] = os.environ.get(out["key_env"]) or ""
    if not out["api_key"]:
        raise RuntimeError(f"{out['key_env']} is not set (needed for provider {name!r}); see .env.example")
    return out


def make_provider(provider: str | None, system: str, tools: list[Tool], model: str | None = None) -> Provider:
    cfg = resolve(provider, model)
    return WIRES[cfg["wire"]](system, tools, cfg)
