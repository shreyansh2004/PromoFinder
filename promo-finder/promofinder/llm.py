"""LLM router with automatic failover across providers and models.

Order of attempts = LLM_PROVIDERS (default "claude,gemini"), each with its own model list.
On a rate limit, timeout/5xx, retired model, auth error or refusal it moves to the next
model/provider. If everything fails, callers fall back to the rule-based path.

Keys and settings live in promo-finder/.env (never committed):
  ANTHROPIC_API_KEY=...                 Claude (official Anthropic SDK)
  CLAUDE_MODELS=claude-opus-5-5         comma list, tried in order
  CLAUDE_EFFORT=medium                  low | medium | high | xhigh | max
  GEMINI_API_KEY=...                    Gemini via its OpenAI-compatible endpoint (LLM_API_KEY also accepted)
  GEMINI_MODELS=gemini-3.5-flash,gemini-flash-latest
  LLM_PROVIDERS=claude,gemini           failover order
Any other OpenAI-compatible API (Groq, OpenRouter, an internal endpoint): set OPENAI_COMPAT_BASE_URL,
OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_MODELS and add "openai_compat" to LLM_PROVIDERS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache" / "llm"))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMUnavailable(RuntimeError):
    pass


class ProviderError(Exception):
    """kind: rate | transient | model | auth | refusal | bad_request"""

    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind


def _models(var: str, default: str) -> list[str]:
    return [m.strip() for m in (os.environ.get(var) or default).split(",") if m.strip()]


# ------------------------------------------------------------------ providers
class ClaudeProvider:
    name = "claude"

    def __init__(self):
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.models = _models("CLAUDE_MODELS", os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"))
        self.effort = os.environ.get("CLAUDE_EFFORT", "medium")
        self.min_gap = float(os.environ.get("CLAUDE_MIN_GAP_S", "0"))
        self.use_fallbacks = True
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def call(self, model: str, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.key, max_retries=0,
                                               timeout=float(os.environ.get("LLM_TIMEOUT_S", "150")))
        sys_prompt = system + ("\nReply with a single JSON object only, no prose, no code fences." if json_mode else "")
        kw = dict(model=model, max_tokens=16000, system=sys_prompt,
                  messages=[{"role": "user", "content": user}], output_config={"effort": self.effort})
        if self.use_fallbacks:  # server-side refusal fallback (routes a declined request to another model)
            kw.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            resp = self._client.beta.messages.create(**kw)
        except anthropic.RateLimitError as e:
            raise ProviderError("rate", str(e))
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise ProviderError("auth", str(e))
        except anthropic.NotFoundError as e:
            raise ProviderError("model", str(e))
        except anthropic.BadRequestError as e:
            if self.use_fallbacks and "fallback" in str(e).lower():
                self.use_fallbacks = False  # account/model without the beta: retry plainly
                return self.call(model, system, user, json_mode)
            raise ProviderError("bad_request", str(e))
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            raise ProviderError("transient", str(e))
        except anthropic.APIStatusError as e:
            raise ProviderError("transient" if e.status_code >= 500 else "bad_request", str(e))
        if resp.stop_reason == "refusal":
            raise ProviderError("refusal", f"refused ({getattr(resp.stop_details, 'category', None)})")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        return text, getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0


class OpenAICompatProvider:
    """Gemini (default) or any OpenAI-compatible endpoint."""

    def __init__(self, name: str, key: str, base_url: str, models: list[str], min_gap: float):
        self.name, self.key, self.base_url, self.models, self.min_gap = name, key, base_url, models, min_gap
        self.json_ok = True
        self.effort = os.environ.get("LLM_REASONING_EFFORT", "")
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.key and self.models)

    def call(self, model: str, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        import openai
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.key, base_url=self.base_url or None, max_retries=0,
                                         timeout=float(os.environ.get("LLM_TIMEOUT_S", "150")))
        kw = dict(model=model, temperature=0.2,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if json_mode and self.json_ok:
            kw["response_format"] = {"type": "json_object"}
        if self.effort:
            kw["reasoning_effort"] = self.effort
        try:
            resp = self._client.chat.completions.create(**kw)
        except openai.RateLimitError as e:
            raise ProviderError("rate", str(e))
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise ProviderError("auth", str(e))
        except openai.NotFoundError as e:
            raise ProviderError("model", str(e))
        except openai.BadRequestError as e:
            msg = str(e).lower()
            if "response_format" in msg and self.json_ok:
                self.json_ok = False
                return self.call(model, system, user, json_mode)
            if "reasoning_effort" in msg and self.effort:
                self.effort = ""
                return self.call(model, system, user, json_mode)
            raise ProviderError("bad_request", str(e))
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            raise ProviderError("transient", str(e))
        except openai.APIStatusError as e:
            raise ProviderError("transient" if e.status_code >= 500 else "bad_request", str(e))
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, (u.prompt_tokens if u else 0) or 0, (u.completion_tokens if u else 0) or 0


def build_providers(model_override: str | None = None) -> list:
    reg = {
        "claude": ClaudeProvider(),
        "gemini": OpenAICompatProvider(
            "gemini", os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY", ""),
            os.environ.get("GEMINI_BASE_URL") or os.environ.get("LLM_BASE_URL") or GEMINI_URL,
            _models("GEMINI_MODELS", "gemini-3.5-flash,gemini-flash-latest"),
            float(os.environ.get("GEMINI_MIN_GAP_S", os.environ.get("LLM_MIN_GAP_S", "4")))),
        "openai_compat": OpenAICompatProvider(
            "openai_compat", os.environ.get("OPENAI_COMPAT_API_KEY", ""), os.environ.get("OPENAI_COMPAT_BASE_URL", ""),
            _models("OPENAI_COMPAT_MODELS", ""), float(os.environ.get("OPENAI_COMPAT_MIN_GAP_S", "0"))),
    }
    order = [p.strip() for p in os.environ.get("LLM_PROVIDERS", "claude,gemini,openai_compat").split(",") if p.strip()]
    provs = [reg[n] for n in order if n in reg and reg[n].configured]
    if model_override:  # e.g. --model gemini-3.5-flash -> that provider + model first
        target = "claude" if model_override.startswith("claude") else ("gemini" if "gemini" in model_override else None)
        for p in provs:
            if p.name == target:
                p.models = [model_override] + [m for m in p.models if m != model_override]
                provs = [p] + [q for q in provs if q is not p]
                break
    return provs


KNOWN_MODELS = {  # offered in the web UI dropdown (plus whatever .env lists)
    "claude": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
    "gemini": ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-pro-preview"],
}


def model_options() -> list[dict]:
    """Models the UI may offer: only providers whose key is configured."""
    out = []
    for p in build_providers():
        seen = []
        for m in p.models + KNOWN_MODELS.get(p.name, []):
            if m not in seen:
                seen.append(m)
        out += [{"provider": p.name, "model": m, "default": i == 0} for i, m in enumerate(seen)]
    return out


# ------------------------------------------------------------------ router
class LLM:
    def __init__(self, model: str | None = None, providers: list | None = None, max_attempts: int = 2):
        self.providers = providers if providers is not None else build_providers(model)
        self.max_attempts = max_attempts
        self.cooldown: dict[str, float] = {}      # provider name -> time when usable again
        self.dead: set[str] = set()                # "provider" or "provider:model" disabled for this run
        self.calls = self.cache_hits = self.tokens_in = self.tokens_out = 0
        self.served_by: dict[str, int] = {}
        self.events: list[str] = []
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}

    @property
    def available(self) -> bool:
        return bool(self.providers)

    @property
    def model(self) -> str:
        c = self._candidates()
        return c[0][1] if c else ""

    def describe(self) -> str:
        if not self.available:
            return "none (rule-based mode)"
        return " -> ".join(f"{p.name}:{p.models[0]}" for p in self.providers if p.models)

    def _candidates(self):
        return [(p, m) for p in self.providers for m in p.models
                if p.name not in self.dead and f"{p.name}:{m}" not in self.dead]

    def _note(self, msg: str):
        self.events.append(msg)
        print(f"  [llm] {msg}", flush=True)

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        cands = self._candidates()
        keys = {(p.name, m): hashlib.sha256(json.dumps([m, system, user, json_mode]).encode()).hexdigest()[:32]
                for p, m in cands}
        for (pn, m), k in keys.items():  # any cached answer from a model in the chain
            cp = CACHE_DIR / f"{k}.json"
            if cp.exists():
                self.cache_hits += 1
                return json.loads(cp.read_text(encoding="utf-8"))["text"]
        for round_ in range(2):
            for p, m in self._candidates():
                if self.cooldown.get(p.name, 0) > time.time():
                    continue
                for attempt in range(self.max_attempts):
                    with self._lock:
                        wait = p.min_gap - (time.time() - self._last.get(p.name, 0))
                        if wait > 0:
                            time.sleep(wait)
                        self._last[p.name] = time.time()
                    try:
                        text, ti, to = p.call(m, system, user, json_mode)
                    except ProviderError as e:
                        if e.kind == "transient" and attempt + 1 < self.max_attempts:
                            time.sleep(3 * (attempt + 1))
                            continue
                        if e.kind == "rate":
                            self.cooldown[p.name] = time.time() + float(os.environ.get("LLM_RATE_COOLDOWN_S", "60"))
                            self._note(f"{p.name} rate-limited, switching provider for a while")
                        elif e.kind == "auth":
                            self.dead.add(p.name)
                            self._note(f"{p.name} key rejected, provider disabled for this run")
                        elif e.kind == "model":
                            self.dead.add(f"{p.name}:{m}")
                            self._note(f"{p.name}:{m} not available, trying next model")
                        else:
                            self._note(f"{p.name}:{m} {e.kind}: {str(e)[:120]}")
                        break
                    with self._lock:
                        self.calls += 1
                        self.tokens_in += ti
                        self.tokens_out += to
                        self.served_by[f"{p.name}:{m}"] = self.served_by.get(f"{p.name}:{m}", 0) + 1
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    (CACHE_DIR / f"{keys.get((p.name, m)) or hashlib.sha256(json.dumps([m, system, user, json_mode]).encode()).hexdigest()[:32]}.json"
                     ).write_text(json.dumps({"model": f"{p.name}:{m}", "text": text}), encoding="utf-8")
                    return text
            # everything cooling down or failed: wait for the soonest cooldown once, then retry
            soon = [t for n, t in self.cooldown.items() if n not in self.dead and t > time.time()]
            if round_ == 0 and soon and min(soon) - time.time() <= 65:
                time.sleep(max(0.0, min(soon) - time.time()))
                continue
            break
        raise LLMUnavailable("all LLM providers failed or are rate-limited")

    def json(self, system: str, user: str, validate=None, repair_rounds: int = 1):
        """Call, parse JSON, optionally validate; one repair round on failure."""
        text = self.complete(system, user)
        for r in range(repair_rounds + 1):
            try:
                obj = parse_json(text)
                if validate:
                    validate(obj)
                return obj
            except LLMUnavailable:
                raise
            except Exception as e:
                if r == repair_rounds:
                    raise
                text = self.complete(system, user + f"\n\nYour previous answer was invalid ({e}). "
                                                    f"Return ONLY valid JSON in the required schema.")


def parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        if start < 0:
            raise
        return json.loads(text[start:text.rfind("}" if text[start] == "{" else "]") + 1])
