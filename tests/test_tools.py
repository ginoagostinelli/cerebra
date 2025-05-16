import pytest
import asyncio
import inspect
from typing import Any, Optional, List, Dict, Type, Union

from pydantic import BaseModel, Field

from cerebra.tools.base import BaseTool, FunctionTool, tool, _python_type_to_json_schema_type, _pydantic_model_to_tool_schema

# --- Test Fixtures and Helper Classes ---


class SimpleInput(BaseModel):
    """Input schema for simple test tools."""

    text: str = Field(description="Some input text.")
    number: int = Field(default=10, description="A number with a default.")


class ComplexInput(BaseModel):
    """Input schema with more complex types."""

    name: str
    items: List[str]
    details: Optional[Dict[str, Any]] = None
    is_active: bool = Field(default=True)


class AnotherModel(BaseModel):
    item_id: int
    value: str


class ModelWithNested(BaseModel):
    name: str
    nested_list: List[AnotherModel]


# --- Synchronous Test Tools (Class-based) ---


class MySyncClassTool(BaseTool):
    """A synchronous tool implemented as a class."""

    args_schema: Type[BaseModel] = SimpleInput

    def __init__(self):
        doc = inspect.getdoc(self) or "A synchronous tool implemented as a class."
        super().__init__(name="MySyncClassTool", description=doc, is_async=False)

    def _run(self, text: str, number: int = 10) -> str:
        return f"ClassTool: {text}-{number}"

    # No _arun needed, BaseTool.arun will use fallback.


class SyncToolRaisesError(BaseTool):
    """A sync tool that always raises an error for testing arun fallback."""

    def __init__(self):
        super().__init__(name="SyncToolRaisesError", description="Raises error.", is_async=False)

    def _run(self) -> str:
        raise ValueError("Sync tool error")

    # No _arun needed.


# --- Asynchronous Test Tools (Class-based) ---


class MyAsyncClassTool(BaseTool):
    """An asynchronous tool implemented as a class."""

    args_schema: Type[BaseModel] = ComplexInput

    def __init__(self):
        doc = inspect.getdoc(self) or "Processes complex input asynchronously."
        super().__init__(name="MyAsyncClassTool", description=doc, is_async=True)

    async def _arun(self, name: str, items: List[str], details: Optional[Dict[str, Any]] = None, is_active: bool = True) -> Dict:
        await asyncio.sleep(0.01)
        return {"name": name, "item_count": len(items), "details_provided": bool(details), "active_status": is_active}

    # No _run needed, BaseTool.run will use fallback.


# --- Function-based Test Tools ---


@tool
def simple_sync_tool(text: str, number: int = 10) -> str:
    """A simple synchronous tool that concatenates text and number."""
    return f"{text}-{number}"


@tool("custom_sync_adder", description="Adds two integers synchronously.")
def custom_name_sync_tool(a: int, b: int) -> int:
    """Docstring for custom_name_sync_tool."""
    return a + b


@tool
async def simple_async_tool(text: str, delay: float = 0.01) -> str:
    """A simple asynchronous tool that introduces a delay."""
    await asyncio.sleep(delay)
    return f"Async: {text}"


@tool("custom_async_multiplier")
async def custom_name_async_tool(a: int, b: int) -> int:
    """Docstring for custom_name_async_tool. Multiplies two numbers asynchronously."""
    await asyncio.sleep(0.01)
    return a * b


@tool
def type_test_tool(
    num: int, flag: bool, val: float, items: List[str], data: Dict[str, Any], opt_num: Optional[int] = None, union_type: Union[int, str] = "default"
):
    return f"{num}-{flag}-{val}-{items}-{data}-{opt_num}-{union_type}"


# Pytest-asyncio provides a function-scoped event loop by default.

# --- FunctionTool Tests ---


def test_simple_sync_tool_execution():
    assert simple_sync_tool.tool.run(text="hello") == "hello-10"
    assert simple_sync_tool.tool.run(text="world", number=20) == "world-20"
    assert simple_sync_tool(text="wrapper") == "wrapper-10"


@pytest.mark.asyncio
async def test_simple_sync_tool_arun_fallback():
    """Test arun fallback for a synchronous function tool."""
    assert await simple_sync_tool.tool.arun(text="hello_arun") == "hello_arun-10"


def test_custom_name_sync_tool_execution():
    tool_instance = custom_name_sync_tool.tool
    assert tool_instance.name == "custom_sync_adder"
    assert tool_instance.run(a=5, b=3) == 8
    assert custom_name_sync_tool(a=1, b=2) == 3


@pytest.mark.asyncio
async def test_simple_async_tool_execution():
    tool_instance = simple_async_tool.tool
    assert tool_instance.is_async
    assert await tool_instance.arun(text="async_world") == "Async: async_world"
    assert await simple_async_tool(text="async_wrapper") == "Async: async_wrapper"


@pytest.mark.asyncio
async def test_simple_async_tool_run_fallback():
    """Test run fallback for an asynchronous function tool.
    This is expected to raise a RuntimeError because asyncio.run()
    cannot be called from an already running event loop (pytest-asyncio's loop).
    """
    with pytest.raises(RuntimeError, match="Cannot call asyncio.run from an already running event loop"):
        simple_async_tool.tool.run(text="async_run_fallback")


def test_function_tool_missing_required_arg():
    with pytest.raises(ValueError, match="Missing required arguments for tool 'simple_sync_tool': text"):
        simple_sync_tool.tool.run(number=50)


def test_function_tool_signature():
    sig = simple_sync_tool.tool.signature()
    assert sig["name"] == "simple_sync_tool"
    assert "text" in sig["parameters"]["properties"]
    assert sig["parameters"]["properties"]["text"]["type"] == "string"
    assert "number" in sig["parameters"]["properties"]
    assert sig["parameters"]["properties"]["number"]["type"] == "integer"
    assert " (Default: 10)" in sig["parameters"]["properties"]["number"]["description"]
    assert "text" in sig["parameters"]["required"]
    assert "number" not in sig["parameters"]["required"]


def test_function_tool_type_conversion_basic():
    tool_instance = custom_name_sync_tool.tool
    assert tool_instance.run(a="5", b="3") == 8


def test_function_tool_type_conversion_comprehensive():
    tool_instance = type_test_tool.tool
    assert (
        tool_instance.run(num="123", flag="true", val="3.14", items='["a", "b"]', data='{"key": "val"}', opt_num="42", union_type="test_str")
        == "123-True-3.14-['a', 'b']-{'key': 'val'}-42-test_str"
    )
    assert tool_instance.run(num=1, flag="no", val="1.0", items="[]", data="{}", union_type=100) == "1-False-1.0-[]-{}-None-100"


def test_function_tool_type_conversion_invalid_bool():
    with pytest.raises(ValueError, match="Error processing argument 'flag'.*Cannot convert string 'maybe' to boolean"):
        type_test_tool.tool.run(num="1", flag="maybe", val="1.0", items="[]", data="{}")


def test_function_tool_type_conversion_invalid_json_list():
    with pytest.raises(
        ValueError, match="Error processing argument 'items' for tool 'type_test_tool': Invalid JSON string for list: Expecting value"
    ):
        type_test_tool.tool.run(num="1", flag="false", val="1.0", items="not a list", data="{}")


def test_function_tool_type_conversion_invalid_json_dict():
    with pytest.raises(ValueError, match="Error processing argument 'data' for tool 'type_test_tool': Invalid JSON string for dict: Expecting value"):
        type_test_tool.tool.run(num="1", flag="true", val="1.0", items="[]", data="not a dict")


def test_function_tool_type_conversion_wrong_json_type():
    # items expects list, gets dict string
    with pytest.raises(ValueError, match="Error processing argument 'items'.*Converted string.*expected list"):
        type_test_tool.tool.run(num="1", flag="true", val="1.0", items='{"key": "value"}', data="{}")

    # data expects dict, gets list string
    with pytest.raises(ValueError, match="Error processing argument 'data'.*Converted string.*expected dict"):
        type_test_tool.tool.run(num="1", flag="true", val="1.0", items="[]", data='["item1"]')


# --- Class-Based Tool Tests ---


def test_sync_class_tool_execution():
    tool_instance = MySyncClassTool()
    assert not tool_instance.is_async
    assert tool_instance.run(text="hello_class") == "ClassTool: hello_class-10"
    assert tool_instance.run(text="world_class", number=25) == "ClassTool: world_class-25"


@pytest.mark.asyncio
async def test_sync_class_tool_arun_fallback():
    tool_instance = MySyncClassTool()
    assert await tool_instance.arun(text="sync_class_arun") == "ClassTool: sync_class_arun-10"


@pytest.mark.asyncio
async def test_sync_tool_arun_fallback_error_propagation():
    tool_instance = SyncToolRaisesError()
    with pytest.raises(ValueError, match="Sync tool error"):
        await tool_instance.arun()


@pytest.mark.asyncio
async def test_async_class_tool_execution():
    tool_instance = MyAsyncClassTool()
    assert tool_instance.is_async
    result = await tool_instance.arun(name="TestApp", items=["item1", "item2"])
    assert result["name"] == "TestApp"
    assert result["item_count"] == 2


@pytest.mark.asyncio
async def test_async_class_tool_run_fallback():
    tool_instance = MyAsyncClassTool()
    with pytest.raises(RuntimeError, match="Cannot call asyncio.run from an already running event loop"):
        tool_instance.run(name="AsyncClassRun", items=["a"])


def test_class_tool_pydantic_validation():
    tool_instance = MySyncClassTool()
    with pytest.raises(ValueError, match="Invalid arguments for tool 'MySyncClassTool' using schema SimpleInput:"):
        tool_instance.run()  # Missing 'text'
    with pytest.raises(ValueError, match="Invalid arguments for tool 'MySyncClassTool' using schema SimpleInput:"):
        tool_instance.run(text="test", number="not_an_int")


def test_class_tool_signature_from_pydantic():
    tool_instance = MySyncClassTool()
    sig = tool_instance.signature()
    assert sig["name"] == "MySyncClassTool"
    params = sig["parameters"]
    assert params["type"] == "object"
    assert "text" in params["properties"]
    assert params["properties"]["text"]["type"] == "string"
    assert "number" in params["properties"]
    assert params["properties"]["number"]["type"] == "integer"
    assert params["properties"]["number"].get("default") == 10
    assert "text" in params["required"]
    assert "number" not in params["required"]


# --- General Tool Decorator Tests ---
def test_tool_decorator_variations():
    @tool
    def func_no_args():
        """Tool with no arguments."""
        return "no_args_executed"

    assert func_no_args.tool.name == "func_no_args"
    assert func_no_args.tool.run() == "no_args_executed"

    @tool("specific_name")
    def func_named(p1: str):
        return f"named_executed: {p1}"

    assert func_named.tool.name == "specific_name"


# --- _python_type_to_json_schema_type helper tests ---
def test_python_type_to_json_schema_type_helper():
    assert _python_type_to_json_schema_type(Optional[int]) == "integer"
    assert _python_type_to_json_schema_type(Optional[Dict[str, Any]]) == "object"
    assert _python_type_to_json_schema_type(Union[str, int]) == "string"  # Default for non-Optional Union
    assert _python_type_to_json_schema_type(List[SimpleInput]) == "array"


# --- _pydantic_model_to_tool_schema helper tests ---
def test_pydantic_model_to_tool_schema_simple_helper():
    schema = _pydantic_model_to_tool_schema(SimpleInput)
    assert schema["type"] == "object"
    assert "text" in schema["required"]
    assert schema["properties"]["number"].get("default") == 10


def test_pydantic_model_to_tool_schema_complex_helper():
    schema = _pydantic_model_to_tool_schema(ComplexInput)
    assert schema["type"] == "object"
    assert "items" in schema["properties"]
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["items"].get("items", {}).get("type") == "string"

    details_schema = schema["properties"]["details"]
    is_object_type = details_schema.get("type") == "object"
    is_object_in_anyof = any(item.get("type") == "object" for item in details_schema.get("anyOf", []))
    assert is_object_type or is_object_in_anyof, "Details schema should represent an object"


def test_pydantic_model_to_tool_schema_with_definitions():
    schema = _pydantic_model_to_tool_schema(ModelWithNested)
    assert schema["type"] == "object"
    assert "nested_list" in schema["properties"]
    assert "$ref" in schema["properties"]["nested_list"]["items"]
    # Pydantic v2 uses '$defs', Pydantic v1 used 'definitions'
    defs_key = "$defs" if "$defs" in schema else "definitions"
    assert defs_key in schema and schema[defs_key]
    ref_name = schema["properties"]["nested_list"]["items"]["$ref"].split("/")[-1]
    assert ref_name in schema[defs_key]


# --- Test for function tool type conversion and schema ---
def test_function_tool_type_conversion_and_schema_relocated():
    @tool("convert_test_relocated")
    def fn_test(a: int, b: float = 3.14, c: bool = False, d: List[int] = [1, 2]):
        return a, b, c, d

    tool_instance: FunctionTool = fn_test.tool
    sig = tool_instance.signature()
    assert sig["parameters"]["properties"]["a"]["type"] == "integer"
    result = tool_instance.run(a="5", b="2.71", c="true", d="[3,4]")
    assert result == (5, 2.71, True, [3, 4])
    with pytest.raises(ValueError, match="Error processing argument 'c'.*Cannot convert string 'not_bool' to boolean"):
        tool_instance.run(a="1", c="not_bool")
