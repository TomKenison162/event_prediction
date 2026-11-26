"""One entry point for every Claude call, with two backends:

  * API key   (ANTHROPIC_API_KEY set)         -> anthropic SDK, Messages API
  * subscription (CLAUDE_CODE_OAUTH_TOKEN set) -> headless Claude Code CLI:
        claude -p --output-format json --json-schema ... --tools ""
    Token comes from `claude setup-token` (valid one year).

The API key wins if both are present (same precedence as Claude Code itself).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from .config import LLM_MODEL


def backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") and shutil.which("claude"):
        return "cli"
    raise RuntimeError("No Claude credentials: set ANTHROPIC_API_KEY, or CLAUDE_CODE_OAUTH_TOKEN "
                       "(from `claude setup-token`) with the Claude Code CLI installed")


def structured(system: str, user: str, schema: dict, effort: str = "high",
               model: str = LLM_MODEL, timeout: int = 900) -> tuple[dict, dict]:
    """Returns (parsed JSON matching schema, metadata)."""
    if backend() == "api":
        return _api(system, user, schema, effort, model)
    return _cli(system, user, schema, model, timeout)


def _api(system, user, schema, effort, model):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=model, max_tokens=16000, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused: {getattr(resp, 'stop_details', None)}")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), {"backend": "api", "model": resp.model, "usage": resp.usage.model_dump()}


def _cli(system, user, schema, model, timeout):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    prompt = f"{system}\n\n---\n\n{user}"
    cmd = ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
           "--model", model, "--tools", ""]
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:      # callers catch RuntimeError -> fallback, never abort the run
        raise RuntimeError(f"claude CLI timed out after {timeout}s") from e
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI failed ({r.returncode}): {r.stdout[-800:]} {r.stderr[-800:]}")
    out = json.loads(r.stdout)
    data = out.get("structured_output")
    if data is None:
        raise RuntimeError(f"claude CLI returned no structured output: {str(out.get('result'))[:800]}")
    return data, {"backend": "cli", "model": model, "cost_usd_estimate": out.get("total_cost_usd"),
                  "usage": out.get("usage")}
