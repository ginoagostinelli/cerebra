import inspect
import functools
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union, get_origin, get_args, get_type_hints

from pydantic import BaseModel, Field

from cerebra.utils.parsing import parse_docstring_params


def _python_type_to_json_schema_type(py_type: Any) -> str:
    """Maps common Python types to JSON Schema type strings for function tool signatures."""
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is str:
        return "string"
    if py_type is bool:
        return "boolean"
    origin = get_origin(py_type)
    if origin is list or py_type is list:
        return "array"
    if origin is dict or py_type is dict:
        return "object"
    if py_type is Any:  # Default 'Any' to string for schema simplicity
        return "string"
    if inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        return "object"  # For Pydantic models used as type hints
    return "string"  # Default for other unhandled types


def _pydantic_model_to_tool_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """Converts a Pydantic V2 model to the JSON schema structure for tool parameters."""
    if not hasattr(pydantic_model, "model_json_schema"):
        raise AttributeError("The provided args_schema model does not have 'model_json_schema'. " "Ensure it's a Pydantic V2 BaseModel.")
    schema = pydantic_model.model_json_schema()
    return {
        "type": schema.get("type", "object"),
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


class BaseTool(ABC):
    """
    Abstract base class for all tools, defining a common interface.

    Attributes:
        name (str): The unique name of the tool.
        description (str): A clear description of what the tool does and when to use it.
        args_schema (Optional[Type[BaseModel]]): Pydantic model defining the input schema (primarily for class-based tools).
        parameters (Dict[str, Dict[str, Any]]): Parameter metadata derived from function signatures/docstrings (for function-based tools).
        is_async (bool): Indicates if the tool's primary execution path is asynchronous (_arun).
        _run_impl_defined (bool): Internal flag, True if _run is implemented by a subclass. Set by subclasses.
        _arun_impl_defined (bool): Internal flag, True if _arun is implemented by a subclass. Set by subclasses.
    """

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    parameters: Dict[str, Dict[str, Any]] = {}
    is_async: bool = False
    _run_impl_defined: bool = False
    _arun_impl_defined: bool = False

    def __init__(self, name: str, description: str, is_async: bool = False):
        self.name = name
        self.description = description
        self.is_async = is_async

    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """Synchronous execution logic. Subclasses must implement this if not async-only."""
        raise NotImplementedError(f"Tool '{self.name}' synchronous execution (_run) not implemented.")

    @abstractmethod
    async def _arun(self, **kwargs) -> Any:
        """Asynchronous execution logic. Subclasses must implement this if not sync-only."""
        raise NotImplementedError(f"Tool '{self.name}' asynchronous execution (_arun) not implemented.")

    def run(self, **kwargs) -> Any:
        """
        Executes the tool's synchronous logic, handling validation and async fallback.

        1. Validates input arguments using `args_schema` if provided.
        2. Calls the `_run` method implemented by the subclass.
        3. If `_run` is not implemented and the tool is async with `_arun` implemented,
           it attempts to run `_arun` blockingly using `asyncio.run`.
        """
        validated_args = kwargs
        if self.args_schema:
            try:
                model_instance = self.args_schema(**kwargs)
                validated_args = model_instance.model_dump()
            except Exception as e:
                raise ValueError(f"Invalid arguments for tool '{self.name}': {e}") from e

        try:
            return self._run(**validated_args)
        except NotImplementedError:
            if self.is_async and hasattr(self, "_arun") and self._arun.__func__ is not BaseTool._arun.__func__:
                try:
                    return asyncio.run(self._arun(**validated_args))
                except RuntimeError as e:
                    # Handle cases where asyncio.run cannot be called (e.g., nested loops)
                    if "cannot be called from a running event loop" in str(e):
                        raise RuntimeError(
                            f"Tool '{self.name}' is async and _run is not implemented. "
                            "Cannot call asyncio.run from an already running event loop. "
                            "Use 'arun' directly or implement a synchronous '_run' method for this context."
                        ) from e
                    raise
            raise

    async def arun(self, **kwargs) -> Any:
        """
        Executes the tool's asynchronous logic, handling validation and sync fallback.

        1. Validates input arguments using `args_schema` if provided.
        2. Calls the `_arun` method implemented by the subclass.
        3. If `_arun` is not implemented and the tool is sync with `_run` implemented,
           it runs `_run` within the event loop's executor.
        """
        validated_args = kwargs
        if self.args_schema:
            try:
                model_instance = self.args_schema(**kwargs)
                validated_args = model_instance.model_dump()
            except Exception as e:
                raise ValueError(f"Invalid arguments for tool '{self.name}': {e}") from e

        try:
            return await self._arun(**validated_args)
        except NotImplementedError:
            if not self.is_async and hasattr(self, "_run") and self._run.__func__ is not BaseTool._run.__func__:
                loop = asyncio.get_event_loop()
                # Run the synchronous method in a thread pool executor to avoid blocking the loop
                return await loop.run_in_executor(None, functools.partial(self._run, **validated_args))
            raise

    def signature(self) -> dict:
        """Generates the tool's signature in a JSON Schema-like format for agent consumption."""
        params_schema: Dict[str, Any]
        if self.args_schema:  # Class-based tools with Pydantic schema
            params_schema = _pydantic_model_to_tool_schema(self.args_schema)
        elif self.parameters:  # Function-based tools derive schema from introspection
            current_properties = {}
            required_params = []
            for param_name, meta in self.parameters.items():
                param_type = meta.get("type", Any)
                is_required = meta.get("required", False)
                description = meta.get("description", "No description available.")
                default_value = meta.get("default", inspect.Parameter.empty)

                prop_entry: Dict[str, Any] = {
                    "type": _python_type_to_json_schema_type(param_type),
                    "description": description,
                }
                if default_value is not inspect.Parameter.empty:
                    # Append default to description for LLM clarity, not standard JSON schema 'default' field
                    prop_entry["description"] += f" (Default: {default_value!r})"

                current_properties[param_name] = prop_entry
                if is_required:
                    required_params.append(param_name)
            params_schema = {"type": "object", "properties": current_properties, "required": sorted(required_params)}
        else:  # Tool takes no arguments
            params_schema = {"type": "object", "properties": {}, "required": []}

        return {
            "name": self.name,
            "description": self.description,
            "parameters": params_schema,
        }


class FunctionTool(BaseTool):
    """
    A concrete implementation of BaseTool that wraps a Python function.
    The `@tool` decorator creates instances of this class.
    """

    func: Callable[..., Any]

    def __init__(self, name: str, description: str, func: Callable[..., Any], parameters: Dict[str, Dict[str, Any]]):
        is_async_func = inspect.iscoroutinefunction(func)
        super().__init__(name=name, description=description, is_async=is_async_func)
        self.func = func
        self.parameters = parameters  # Parameter metadata derived via introspection

        if self.is_async:
            self._arun_impl_defined = True
        else:
            self._run_impl_defined = True

    def _run(self, **kwargs) -> Any:
        """Overrides BaseTool._run to execute the wrapped synchronous function."""
        if self.is_async:
            # This state should ideally be handled by BaseTool.run's fallback logic
            raise RuntimeError(f"Cannot synchronously run async FunctionTool '{self.name}'.")
        validated_args = self._validate_and_convert_args(**kwargs)
        return self.func(**validated_args)

    async def _arun(self, **kwargs) -> Any:
        """Overrides BaseTool._arun to execute the wrapped asynchronous function."""
        if not self.is_async:
            # This state should ideally be handled by BaseTool.arun's fallback logic
            raise RuntimeError(f"Cannot asynchronously run sync FunctionTool '{self.name}'.")
        validated_args = self._validate_and_convert_args(**kwargs)
        return await self.func(**validated_args)

    def _validate_and_convert_args(self, **kwargs) -> Dict[str, Any]:
        """
        Validates and converts arguments for function-based tools based on parameter metadata.
        Provides basic string-to-type conversions (e.g., "true"->True, "123"->123, json string -> list/dict).
        More complex validation relies on the wrapped function's own logic.
        """
        validated_args = {}
        missing_required = []

        for param_name, meta in self.parameters.items():
            expected_type = meta.get("type", Any)
            is_required = meta.get("required", False)
            default_value = meta.get("default", inspect.Parameter.empty)

            if param_name in kwargs:
                value = kwargs[param_name]
                try:
                    # Attempt basic type conversions if input is string and expected type is different
                    if isinstance(value, str) and expected_type is not str:
                        if expected_type is bool:
                            lower_val = value.lower()
                            if lower_val in ["true", "1", "yes", "y"]:
                                value = True
                            elif lower_val in ["false", "0", "no", "n"]:
                                value = False
                            else:
                                raise ValueError(f"Cannot convert string '{value}' to boolean.")
                        elif expected_type is int:
                            value = int(value)
                        elif expected_type is float:
                            value = float(value)
                        elif get_origin(expected_type) is list:  # Expect JSON array string
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                raise ValueError(f"Cannot convert string '{value}' to list. Expected JSON array string.")
                        elif get_origin(expected_type) is dict:  # Expect JSON object string
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                raise ValueError(f"Cannot convert string '{value}' to dict. Expected JSON object string.")

                    validated_args[param_name] = value  # Assign the potentially converted value

                except (ValueError, TypeError) as e:
                    # Add context to conversion errors
                    raise type(e)(f"Error processing argument '{param_name}' for tool '{self.name}': {e}") from e

            else:  # Argument not provided by caller
                if is_required and default_value is inspect.Parameter.empty:
                    missing_required.append(param_name)
                elif default_value is not inspect.Parameter.empty:
                    validated_args[param_name] = default_value
                # If not required and no default, implicitly None or handled by function

        if missing_required:
            raise ValueError(f"Missing required arguments for tool '{self.name}': {', '.join(missing_required)}")
        return validated_args


def tool(name_or_func: Union[str, Callable[..., Any], None] = None, description: Optional[str] = None):
    """
    Decorator to register a Python function as a callable Tool instance.

    Derives tool name, description, and parameters from the function signature,
    type hints, and docstring unless overridden by decorator arguments.

    Attaches a `.tool` attribute to the returned wrapper function
    """
    if callable(name_or_func) and description is None:  # Standard @tool usage
        func = name_or_func
        tool_name_str = func.__name__
        docstring = inspect.getdoc(func) or ""
        # Default description is the first paragraph of the docstring
        tool_description_str = docstring.split("\n\n", 1)[0].strip() if docstring else f"Tool '{tool_name_str}'"
        return _create_function_tool_wrapper(func, tool_name_str, tool_description_str)
    else:  # Decorator with arguments: @tool("name"), @tool(description="..."), or @tool("name", description="...")
        actual_name_arg: Optional[str] = name_or_func if isinstance(name_or_func, str) else None
        actual_desc_arg: Optional[str] = description

        def decorator(func: Callable[..., Any]):
            # Determine final name: decorator arg > function name
            final_name = actual_name_arg or func.__name__
            docstring = inspect.getdoc(func) or ""
            # Determine final description: decorator arg > docstring summary > default
            final_description = (
                actual_desc_arg if actual_desc_arg is not None else (docstring.split("\n\n", 1)[0].strip() if docstring else f"Tool '{final_name}'")
            )
            return _create_function_tool_wrapper(func, final_name, final_description)

        return decorator


def _create_function_tool_wrapper(func: Callable[..., Any], tool_name: str, tool_description: str):
    """Internal helper to perform function introspection and create the FunctionTool instance and wrapper."""
    param_docs = parse_docstring_params(inspect.getdoc(func))
    sig = inspect.signature(func)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    parameters_metadata: Dict[str, Dict[str, Any]] = {}
    for param_name, param in sig.parameters.items():
        # Exclude *args and **kwargs from the generated tool signature
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        raw_type_hint = type_hints.get(param_name, Any)
        actual_type = raw_type_hint
        is_optional_type = False  # Is the type hint Optional[T] or Union[T, None]?

        origin = get_origin(raw_type_hint)
        if origin is Union:
            union_args = get_args(raw_type_hint)
            is_optional_type = type(None) in union_args
            if is_optional_type:
                # Simplify Optional[T] to just T for the 'type' metadata
                non_none_args = [t for t in union_args if t is not type(None)]
                if len(non_none_args) == 1:
                    actual_type = non_none_args[0]

        if actual_type is Any:  # Default 'Any' type to string for schema generation
            actual_type = str

        has_default = param.default is not inspect.Parameter.empty
        # A parameter is required if it has no default value AND it's not an Optional type.
        is_required = not has_default and not is_optional_type

        param_info: Dict[str, Any] = {
            "type": actual_type,
            "description": param_docs.get(param_name, f"Parameter '{param_name}'."),
            "required": is_required,
        }
        if has_default:
            param_info["default"] = param.default
        parameters_metadata[param_name] = param_info

    tool_instance = FunctionTool(name=tool_name, description=tool_description, func=func, parameters=parameters_metadata)

    # Create the callable wrapper that users interact with via the decorator.
    # This wrapper delegates calls to the tool_instance's run/arun methods.
    if tool_instance.is_async:

        @functools.wraps(func)
        async def async_tool_wrapper(**kwargs):
            return await tool_instance.arun(**kwargs)

        # Attach the tool instance to the wrapper for discovery (e.g., by Agent)
        async_tool_wrapper.tool = tool_instance  # type: ignore[attr-defined]
        return async_tool_wrapper
    else:

        @functools.wraps(func)
        def sync_tool_wrapper(**kwargs):
            return tool_instance.run(**kwargs)

        # Attach the tool instance to the wrapper
        sync_tool_wrapper.tool = tool_instance  # type: ignore[attr-defined]
        return sync_tool_wrapper
