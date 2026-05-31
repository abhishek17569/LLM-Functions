# LLM Functions

Turn a Python function signature and docstring into an LLM-backed implementation.

```python
@llm_function(model="openai/gpt-4o-mini")
def is_man(text: str) -> bool:
    """Decide whether the text is talking about a man."""

is_man("a fellow named John walked in")  # -> True
```

The decorator reads the function's name, docstring, parameter types, and
return-type annotation, builds a JSON-Schema for the output, and asks the
model to fill it in via native function-calling. The result is validated
into the declared return type. Streaming, tool-calling, declared exceptions,
result caching, and deterministic replay-from-fixture are all supported.

## Install

```bash
uv add llm-functions   # or: pip install llm-functions
```

You also need a configured LiteLLM provider (an API key for the model you
pick — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

## Examples

The three examples below match the ones exercised by `tests/test_e2e.py`,
which runs them in `cache="replay"` mode against committed JSON fixtures —
no network, no API key needed. To actually call the model from your own
code, drop `cache="replay"` (or set it to `"on"` / `"off"`) and configure
the appropriate provider credential.

### 1. Boolean classifier

```python
from llm_functions import llm_function

@llm_function(cache="replay", model="openai/gpt-4o-mini")
def is_man(text: str) -> bool:
    """Decide whether the text is talking about a man.

    Args:
        text: A short sentence about a person.

    Returns:
        True if the subject is a man, False otherwise.
    """

assert is_man("a fellow named John walked in") is True
```

### 2. Declared exceptions

Functions can declare exceptions in a `Raises:` section. The decorator
synthesises a `raise_<ExcName>` tool the model can invoke when it cannot
satisfy the contract; the call is converted back into the declared Python
exception with the model's reason as the message.

```python
from llm_functions import llm_function

class NameNotFoundError(Exception):
    """No name was found in the text."""

@llm_function(cache="replay", model="openai/gpt-4o-mini")
def find_names(text: str) -> list[str]:
    """Extract every personal name from the text.

    Args:
        text: A passage that may contain names.

    Returns:
        Names in order of first appearance.

    Raises:
        NameNotFoundError: If the text contains no names at all.
    """

assert find_names("Alice and Bob went home.") == ["Alice", "Bob"]
```

### 3. Streaming

Annotate the return type as `AsyncIterator[T]` and the wrapper streams
results as they arrive. `T = str` streams text deltas; `T = SomeModel`
streams partial Pydantic models; `T = int` (or any JSON-able type) streams
list items.

```python
import asyncio
from collections.abc import AsyncIterator

from llm_functions import llm_function

@llm_function(cache="replay", model="openai/gpt-4o-mini")
async def greet(name: str) -> AsyncIterator[str]:
    """Stream a friendly greeting.

    Args:
        name: The recipient's name.

    Returns:
        Sentence fragments that concatenate to the full greeting.
    """

async def main() -> None:
    pieces: list[str] = []
    async for piece in await greet("Alice"):
        pieces.append(piece)
    print("".join(pieces))

asyncio.run(main())
```

## Docstring reference — every supported section

A single Google-style example that exercises every section the parser
understands. Aliases are accepted (e.g. `Return` ≡ `Returns`,
`Param`/`Parameter`/`Parameters` ≡ `Args`, `Note` ≡ `Notes`). Sphinx
(`:param x:` / `:returns:` / `:raises Exc:`) and NumPy
(`Parameters\n----------`) styles are also auto-detected.

```python
from pydantic import BaseModel, Field

from llm_functions import llm_function, llm_tool


class ActionItem(BaseModel):
    owner: str = Field(description="Person responsible for the action.")
    description: str = Field(description="What needs to be done.")
    due: str | None = Field(default=None, description="ISO-8601 due date.")


class NoActionItemsError(Exception):
    """Raised when the transcript contains no actionable items."""


@llm_tool
def lookup_employee(name: str) -> str:
    """Return the canonical full name and team of an employee."""
    ...


@llm_function(tools=[lookup_employee])
def extract_action_items(transcript: str, max_items: int = 10) -> list[ActionItem]:
    """Extract action items from a meeting transcript.

    Context:
        An "action item" is a concrete task assigned to a named person with
        a clear deliverable. Status updates, opinions, and rhetorical
        questions are not action items.

    Args:
        transcript: Free-form meeting transcript. May contain speaker tags
            like ``Alice:`` or timestamps; treat both as noise.
        max_items: Hard cap on returned items. If more candidates exist,
            return the most prominent ones.

    Returns:
        A list of ``ActionItem`` of length 0..``max_items``, ordered by
        appearance in the transcript.

    Raises:
        NoActionItemsError: If the transcript contains no actionable items
            at all (e.g. a status-only meeting).

    Examples:
        >>> extract_action_items("Alice: I'll send the deck by Friday.")
        [ActionItem(owner='Alice', description='Send the deck', due='Friday')]

        Conversational transcripts work the same way; speaker tags are
        stripped before extraction.

    Constraints:
        - Never invent owners not named in the transcript.
        - Honorifics (Mr., Dr.) are not part of the owner's name.
        - Merge duplicates that refer to the same task.

    Tools:
        lookup_employee: Use ONLY to disambiguate two same-first-name
            owners. Do not use for primary extraction.

    Format:
        Return strictly conforming JSON. No prose, no Markdown.

    Notes:
        Prefer precision over recall — when uncertain whether something is
        an action item, omit it.

    Model: openai/gpt-4o-mini
    Cache: on
    """
```

Section reference:

| Section | Purpose |
|---|---|
| `Task:` (or the leading paragraph) | Core instruction |
| `Context:` | Background and definitions |
| `Args:` | Per-parameter description (types come from annotations) |
| `Returns:` | Return-value description |
| `Raises:` | LLM-actionable exceptions — each becomes a `raise_<Exc>` tool |
| `Examples:` | Doctest blocks become few-shots; prose is appended verbatim |
| `Constraints:` | Hard rules enforced through the repair loop |
| `Tools:` | Per-tool guidance (the allow-list still lives on the decorator) |
| `Format:` | Extra output-format hints |
| `Notes:` | Soft guidance — tone, precision/recall, style |
| `Model:` | Per-function model override (logs INFO when overriding global) |
| `Cache:` | Per-function cache policy `on`/`off`/`replay` |

## Tools

Pass any callable in `tools=[...]` and the model can invoke it during the
run. Tools are validated against a JSON-Schema derived from their signature.

```python
from llm_functions import llm_function, llm_tool

@llm_tool
def search_web(query: str) -> str:
    """Search the web for ``query`` and return the top result."""
    ...

@llm_function(tools=[search_web])
def answer(question: str) -> str:
    """Answer the user's question, using ``search_web`` when needed."""
```

To expose every `@llm_tool`-decorated function in the process, pass
`tools="*"` and call `configure(allow_all_tools=True)` first — the
configure call is a deliberate guardrail so a misplaced `tools="*"` never
silently exposes everything.

## Caching modes

Each call's result is keyed by the function source, model, prompt, tool
schemas, and arguments.

| `cache=` | Behaviour |
|---|---|
| `"on"` (default) | Read fixture if present; otherwise call the model and write the result. |
| `"off"` | Always call the model. |
| `"replay"` | Read fixture only; raise `ReplayMissError` if missing. Used for deterministic test runs. |

Configure the cache directory globally:

```python
from llm_functions import configure
configure(cache_dir="./.ai_function_cache",
          replay_fixtures_dir="./tests/fixtures/llm_function")
```

## `configure(...)` reference

Process-global defaults. Repeated calls merge — fields you don't pass keep
their current value. Per-decorator and per-call overrides win over these.

| Key | Type | Default | Notes |
|---|---|---|---|
| `model` | `str` | `"openai/gpt-4o-mini"` | LiteLLM-style `provider/model` string. |
| `temperature` | `float ≥ 0` | `0.0` | Some models (GPT-5) reject `0.0`; the framework auto-drops it on retry. |
| `cache` | `"on" \| "off" \| "replay"` | `"on"` | See "Caching modes" above. |
| `cache_dir` | `str \| None` | `XDG_CACHE_HOME` | Where `diskcache` stores results. |
| `replay_fixtures_dir` | `str` | `"tests/fixtures/llm_function"` | JSON fixtures for `cache="replay"`. |
| `max_repairs` | `int ≥ 0` | `2` | Max retries when the model's structured output fails validation. |
| `max_tool_iterations` | `int ≥ 0` | `10` | Cap on tool-call rounds in a single function call. |
| `timeout` | `float > 0` | `60.0` | Per-call HTTP timeout. |
| `allow_all_tools` | `bool` | `False` | Required for `tools="*"`. |
| `providers` | `dict[str, ProviderConfig \| dict]` | `None` | Per-provider keys, gateway URLs, headers — see "Provider credentials". |
| `log_level` | `"DEBUG" \| "INFO" \| ...` or `int` | not set | Installs a dedicated handler on the `llm_functions` logger; `propagate=False` so root-level libraries (LiteLLM, httpx) stay quiet. |
| `log_stream` | file-like | `sys.stderr` | Optional stream for the installed log handler. Pair with `log_level=`. |

## Provider credentials

`llm_function` reads credentials in two ways. By default it relies on the
provider-specific environment variables LiteLLM expects (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.). For apps that load secrets
from a vault, route through an OpenAI-compatible gateway (TrueFoundry,
Portkey, custom OAuth proxies), or need to refresh JWTs, configure each
provider explicitly:

```python
from llm_functions import configure, ProviderConfig

configure(providers={
    "openai":    ProviderConfig(api_key="sk-..."),
    "anthropic": ProviderConfig(api_key="sk-ant-..."),
})
```

Each `ProviderConfig` carries:

| Field | Purpose |
|---|---|
| `api_key` | A string or a zero-arg `Callable[[], str]`. Callables are invoked **once per LLM call**, so refresh-token / JWT flows just plug in. |
| `api_base` | Override the upstream URL — point at a gateway, an Azure deployment, an Ollama instance, etc. |
| `extra_headers` | Forwarded verbatim to LiteLLM. Useful for tenant/project metadata required by gateways. |

Provider keys match the prefix in `model=` (`openai/gpt-4o` → `openai`,
`anthropic/claude-sonnet-4-6` → `anthropic`). Repeated `configure` calls
**merge** per-provider entries; pass `providers=None` to clear them all.
Providers without an entry fall through to environment variables, so you
can mix configured and env-based providers.

### TrueFoundry / OpenAI-compatible gateway with JWT refresh

```python
import time

import jwt_helper                                # your auth client
from llm_functions import configure, ProviderConfig

def fresh_jwt() -> str:                          # called per LLM call
    return jwt_helper.get_token()                # rotates automatically

configure(providers={
    "openai": ProviderConfig(
        api_key=fresh_jwt,
        api_base="https://llm-gateway.truefoundry.com/api/inference/openai",
        extra_headers={"X-TFY-METADATA": "tenant=foo,project=bar"},
    ),
})
```

The model string still uses the `openai/...` prefix because the gateway
speaks the OpenAI wire format; `api_base` redirects the request to your
gateway and the JWT is supplied as the bearer token.

## Inspecting the prompt

Two ways to see exactly what was sent to the model.

**Per-call, ad-hoc.** Pass `debug=True` to print the rendered system /
user messages, the resolved model, the tool list, and the output schema
to `stderr`:

```python
poem = await write_poem("Ada", "a curious mathematician", debug=True)
```

```
========================================================================
llm_function debug: write_poem
========================================================================
model:       openai/openai-main/gpt-5
temperature: 0.0
cache:       on
api_base:    https://llm-gateway.truefoundry.com/api/inference/openai

--- system ---
You are the implementation of the Python function `write_poem`.
Follow its docstring contract precisely.

Task:
Compose a short, warm poem about a person.
...

--- user ---
Call write_poem with these arguments:
{
  "name": "Ada",
  "description": "a curious mathematician"
}

--- tools ---
emit_final_answer

--- output schema ---
{ ... }
========================================================================
```

**Always-on, structured.** Call `configure(log_level="DEBUG")` once at
startup. This installs a dedicated handler on the `llm_functions` logger
and disables propagation to the root logger — so you get the full
request payload (messages, tools, sampling params) on every call,
including param-drop retries, **without** dragging LiteLLM, httpx, or
urllib3 logs along for the ride. `api_key` is omitted and
`extra_headers` values are redacted so dumps from production logs don't
leak credentials.

```python
from llm_functions import configure
configure(log_level="DEBUG")                  # only llm_function logs
configure(log_level="DEBUG", log_stream=open("/tmp/llm_function.log", "w"))
```

If you'd rather wire the logger yourself (e.g. through your app's
`dictConfig`), set the level on `logging.getLogger("llm_functions")`,
attach a handler, and set `propagate = False`. The library writes its
DEBUG records there regardless of who installed the handler.

## Unsupported parameters and auto-retry

Different models accept different parameters: GPT-5 only allows
`temperature=1`, some Anthropic models reject `response_format`, certain
gateways strip `parallel_tool_calls`. When the upstream provider raises
`UnsupportedParamsError`, `llm_function` parses the offending parameter
names out of the error and retries the call once with those parameters
dropped — so the same code keeps working when you point it at a stricter
model. The retry is logged at `INFO` so you can audit what was dropped.
If the error names something the framework doesn't recognise, it's
surfaced unchanged rather than retried blindly.

## Per-call overrides

Any of `cache`, `model`, `temperature`, `max_repairs`, `timeout` may be
passed at call time and override the decorator-level value. `debug=True`
prints the rendered prompt for that single call (see "Inspecting the
prompt" above).

```python
is_man("…", model="anthropic/claude-3-5-sonnet-latest", temperature=0.0)
is_man("…", debug=True)
```

## Public API

```python
from llm_functions import (
    llm_function, llm_tool, configure, ProviderConfig,
    LLMFunctionError, ConfigurationError, OutputValidationError,
    ProviderError, ReplayMissError, ToolExecutionError, ToolIterationError,
)
```

## License

MIT.
