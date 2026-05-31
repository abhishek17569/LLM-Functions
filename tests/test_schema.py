"""Schema-builder fixtures.

Each schema is validated with ``jsonschema.Draft202012Validator.check_schema``
to catch structural defects in the generator.

We intentionally omit ``from __future__ import annotations`` so that
function-local Pydantic models and Enum types resolve to their real class
objects in :class:`inspect.Signature`. Production callers similarly run the
decorator at function-definition time, when annotations are concrete.
"""

from enum import Enum
from typing import Literal, Union

import jsonschema
from pydantic import BaseModel, Field

from llmfunctionkit._schema import build_input_schema, build_output_schema


def _check(schema: dict[str, object]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


def test_primitive_input_and_output() -> None:
    def fn(x: int, y: float, name: str, flag: bool) -> str:
        """Take primitives.

        Args:
            x: an integer
            y: a float
            name: a name
            flag: a boolean
        """
        return name

    inp = build_input_schema(fn)
    out = build_output_schema(fn)
    _check(inp)
    _check(out)
    assert inp["type"] == "object"
    props = inp["properties"]
    assert isinstance(props, dict)
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "number"
    assert props["name"]["type"] == "string"
    assert props["flag"]["type"] == "boolean"
    assert props["x"]["description"] == "an integer"
    assert out["type"] == "string"


def test_list_input() -> None:
    def fn(items: list[int]) -> int:
        return sum(items)

    inp = build_input_schema(fn)
    _check(inp)
    items = inp["properties"]["items"]
    assert items["type"] == "array"
    assert items["items"] == {"type": "integer"}


def test_dict_input() -> None:
    def fn(mapping: dict[str, int]) -> int:
        return sum(mapping.values())

    inp = build_input_schema(fn)
    _check(inp)
    mapping = inp["properties"]["mapping"]
    assert mapping["type"] == "object"
    assert mapping["additionalProperties"] == {"type": "integer"}


def test_optional_input() -> None:
    def fn(name: str | None = None) -> str:
        return name or ""

    inp = build_input_schema(fn)
    _check(inp)
    name = inp["properties"]["name"]
    types = name.get("anyOf") or name.get("type")
    assert types is not None


def test_union_input() -> None:
    def fn(value: Union[int, str]) -> str:  # noqa: UP007
        return str(value)

    inp = build_input_schema(fn)
    _check(inp)
    val = inp["properties"]["value"]
    assert "anyOf" in val
    type_set = {entry.get("type") for entry in val["anyOf"]}
    assert {"integer", "string"} <= type_set


def test_literal_input() -> None:
    def fn(mode: Literal["fast", "slow"]) -> str:
        return mode

    inp = build_input_schema(fn)
    _check(inp)
    mode = inp["properties"]["mode"]
    assert mode["enum"] == ["fast", "slow"]


def test_enum_input() -> None:
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    def fn(c: Color) -> str:
        return c.value

    inp = build_input_schema(fn)
    _check(inp)
    c = inp["properties"]["c"]
    assert c.get("enum") == ["red", "blue"] or c.get("type") == "string"


def test_pydantic_model_with_field_description() -> None:
    class Person(BaseModel):
        name: str = Field(..., description="Full name")
        age: int = Field(..., description="Age in years", ge=0)

    def fn(p: Person) -> str:
        return p.name

    inp = build_input_schema(fn)
    _check(inp)
    p = inp["properties"]["p"]
    props = p["properties"]
    assert props["name"]["description"] == "Full name"
    assert props["age"]["description"] == "Age in years"
    assert props["age"]["minimum"] == 0


def test_nested_pydantic_model() -> None:
    class Address(BaseModel):
        street: str = Field(..., description="Street name")

    class Customer(BaseModel):
        name: str
        address: Address

    def fn(c: Customer) -> str:
        return c.name

    inp = build_input_schema(fn)
    _check(inp)
    c = inp["properties"]["c"]
    addr = c["properties"]["address"]
    assert addr["properties"]["street"]["description"] == "Street name"


def test_list_of_pydantic_model_input() -> None:
    class Item(BaseModel):
        sku: str = Field(..., description="Stock-keeping unit")

    def fn(items: list[Item]) -> int:
        return len(items)

    inp = build_input_schema(fn)
    _check(inp)
    items = inp["properties"]["items"]
    assert items["type"] == "array"
    assert items["items"]["properties"]["sku"]["description"] == "Stock-keeping unit"


def test_returns_description_merged_from_docstring() -> None:
    def fn() -> int:
        """Compute it.

        Returns:
            The number of widgets in the warehouse.
        """
        return 0

    out = build_output_schema(fn)
    _check(out)
    assert out["type"] == "integer"
    assert out["description"] == "The number of widgets in the warehouse."


def test_args_description_merged_from_docstring() -> None:
    def fn(slug: str) -> str:
        """Look it up.

        Args:
            slug: The slug to query.
        """
        return slug

    inp = build_input_schema(fn)
    _check(inp)
    assert inp["properties"]["slug"]["description"] == "The slug to query."


def test_pydantic_field_description_wins_over_docstring() -> None:
    class Payload(BaseModel):
        text: str = Field(..., description="Pydantic-side description")

    def fn(p: Payload) -> str:
        """Run.

        Args:
            p: Docstring-side description.
        """
        return p.text

    inp = build_input_schema(fn)
    _check(inp)
    p = inp["properties"]["p"]
    assert p["properties"]["text"]["description"] == "Pydantic-side description"


def test_pydantic_model_return() -> None:
    class Result(BaseModel):
        ok: bool
        message: str = Field(..., description="Status message")

    def fn() -> Result:
        return Result(ok=True, message="hi")

    out = build_output_schema(fn)
    _check(out)
    assert out["type"] == "object"
    assert out["properties"]["message"]["description"] == "Status message"
