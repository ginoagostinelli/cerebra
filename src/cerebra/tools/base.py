import inspect
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, get_type_hints, Union, get_origin, get_args

from cerebra.utils.parsing import parse_docstring_params


def _python_type_to_json_schema_type(py_type: Any) -> str:
    """Maps Python types to JSON Schema type strings"""
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is str:
        return "string"
    if py_type is bool:
        return "boolean"
    if py_type is list or get_origin(py_type) is list:
        return "array"
    if py_type is dict or get_origin(py_type) is dict:
        return "object"

    return "string"  # default to string for simplicity.


@dataclass
class Tool:
    """Represents a callable tool"""

    name: str
    description: str
    func: Callable[..., Any]
    # parameter metadata: {'param_name': {'type': type_obj, 'description': str, 'required': bool, 'default': value}}
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __call__(self, **kwargs) -> Any:
        """
        Executes the tool's function with validated and typed arguments.

        Args:
            **kwargs: Keyword arguments for the tool function.

        Returns:
            The result of the tool's function execution.
        """
        validated_args = {}
        missing_required = []

        for param_name, meta in self.parameters.items():
            expected_type = meta.get("type", str)  # Default to string if type info missing
            is_required = meta.get("required", False)
            default_value = meta.get("default", inspect.Parameter.empty)

            if param_name in kwargs:
                value = kwargs[param_name]

                if isinstance(value, expected_type):
                    validated_args[param_name] = value
                    continue

                # Attempt common conversions
                if isinstance(value, str):
                    try:
                        if expected_type is bool:
                            lower_val = value.lower()
                            if lower_val in ["true", "1", "yes", "y"]:
                                validated_args[param_name] = True
                            elif lower_val in ["false", "0", "no", "n"]:
                                validated_args[param_name] = False
                            else:
                                raise ValueError(f"Cannot convert string '{value}' to boolean.")
                        elif expected_type in (int, float):
                            validated_args[param_name] = expected_type(value)
                        elif expected_type is str:
                            validated_args[param_name] = value
                        else:
                            raise TypeError(f"Automatic string conversion not supported for type {expected_type.__name__}.")
                        continue  # Skip to next param if conversion successful
                    except ValueError as e:
                        raise ValueError(
                            f"Argument '{param_name}' received string '{value}', but failed to convert to {expected_type.__name__}: {e}"
                        ) from e
                    except TypeError as e:
                        raise TypeError(
                            f"Argument '{param_name}' received string '{value}', encountered type issue during conversion to {expected_type.__name__}: {e}"
                        ) from e

                raise TypeError(f"Argument '{param_name}' expected type {expected_type.__name__} but got {type(value).__name__} ({value!r})")

            else:
                # Argument not provided
                if is_required and default_value is inspect.Parameter.empty:
                    missing_required.append(param_name)
                elif default_value is not inspect.Parameter.empty:
                    validated_args[param_name] = default_value
                # If not required and no default, it's implicitly None or handled by function default

        if missing_required:
            raise ValueError(f"Missing required arguments: {', '.join(missing_required)}")

        try:
            # print(f"DEBUG: Calling {self.name} with args: {validated_args}")
            return self.func(**validated_args)
        except Exception as e:
            print(f"Error executing tool '{self.name}': {e}")
            raise

    def signature(self) -> dict:
        """
        Generates the tool's signature in a JSON Schema-like format,

        Returns:
            A dictionary representing the tool's signature.
        """
        properties = {}
        required_params = []

        for param_name, meta in self.parameters.items():
            # Extract the core type if Optional[T] or Union[T, None]
            param_type = meta.get("type", Any)
            is_required = meta.get("required", False)
            description = meta.get("description", "No description available")
            default_value = meta.get("default", inspect.Parameter.empty)

            prop_entry = {
                "type": _python_type_to_json_schema_type(param_type),
                "description": description,
            }

            # Note: JSON schema standard doesn't put default inside properties usually, but it's helpful for LLM description.
            if default_value is not inspect.Parameter.empty:
                prop_entry["description"] += f" (Default: {default_value!r})"
                # prop_entry["default"] = default_value  # add default to the schema

            properties[param_name] = prop_entry

            if is_required:
                required_params.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": sorted(required_params),  # Sort for consistency
            },
        }


def tool(name: Optional[str] = None):
    """
    Decorator to register a Python function as a Tool.

    Args:
        name: Optional explicit name for the tool. If None, the function's name is used.
    """

    def decorator(func: Callable[..., Any]):
        tool_name = name or func.__name__
        docstring = inspect.getdoc(func) or ""
        # Use the first line or paragraph as the main description
        summary = docstring.split("\n\n", 1)[0] if docstring else f"Tool '{tool_name}'"
        param_docs = parse_docstring_params(docstring)
        sig = inspect.signature(func)
        try:
            # Use globalns/localns if needed for forward references, though often not necessary here
            type_hints = get_type_hints(func)
        except Exception as e:
            # print(f"Warning: Could not get type hints for {tool_name}: {e}. Using defaults.")
            type_hints = {}

        parameters_metadata: Dict[str, Dict[str, Any]] = {}
        for param_name, param in sig.parameters.items():
            # Ignore *args and **kwargs parameters for the tool signature
            if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
                print(f"Warning: Ignoring {param.kind.description} parameter '{param_name}' in tool '{tool_name}' signature.")
                continue

            raw_type_hint = type_hints.get(param_name, Any)

            # Handle Optional[T] and Union[T, None]
            origin = get_origin(raw_type_hint)
            is_optional = False
            actual_type = raw_type_hint

            if origin is Union:
                args = get_args(raw_type_hint)
                non_none_args = [t for t in args if t is not type(None)]
                if len(non_none_args) == 1 and len(args) == 2:  # Checks for Union[T, None] or Optional[T]
                    actual_type = non_none_args[0]
                    is_optional = True
                # Note: More complex Unions (Union[int, str]) are harder to represent directly in basic JSON schema 'type'. For now, defaults to 'string'.
                # Consider using 'anyOf' in JSON schema for complex unions.
            elif raw_type_hint is Any:
                actual_type = str

            has_default = param.default is not inspect.Parameter.empty
            is_required_in_signature = not has_default and not is_optional

            param_info = {
                "type": actual_type,
                "description": param_docs.get(param_name, "No description available."),
                "required": is_required_in_signature,
            }
            if has_default:
                param_info["default"] = param.default

            parameters_metadata[param_name] = param_info

        tool_instance = Tool(name=tool_name, description=summary, func=func, parameters=parameters_metadata)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args:
                raise TypeError(f"Tool '{tool_instance.name}' must be called with keyword arguments, not positional arguments.")
            return tool_instance(**kwargs)

        wrapper.tool = tool_instance

        return wrapper

    return decorator
