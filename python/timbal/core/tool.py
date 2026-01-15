import inspect
from collections.abc import Callable
from functools import cached_property
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from pydantic import BaseModel, SkipValidation, computed_field, model_validator

from ..utils import create_model_from_handler
from .runnable import Runnable


def analyze_callable_signature(callable_fn: Callable[..., Any]) -> dict[str, Any]:
    """
    Analyze a callable function's signature and return detailed information.
    
    Args:
        callable_fn: The callable function to analyze
        
    Returns:
        Dictionary containing:
        - name: Function name
        - parameters: List of parameter names
        - required_parameters: List of required parameter names
        - optional_parameters: List of optional parameter names
        - has_varargs: Whether function accepts *args
        - has_varkw: Whether function accepts **kwargs
        - has_return_annotation: Whether return type is annotated
        - return_annotation: Return type annotation if present
        - is_method: Whether this is a bound method
    """
    if not callable(callable_fn):
        raise ValueError("Provided object is not callable")
    
    try:
        argspec = inspect.getfullargspec(callable_fn)
        parameters = argspec.args
        defaults = argspec.defaults or []
        num_required = len(parameters) - len(defaults)
        required_params = parameters[:num_required]
        optional_params = parameters[num_required:]
        
        return_annotation = argspec.annotations.get("return", None)
        
        is_method = inspect.ismethod(callable_fn) or (
            inspect.isfunction(callable_fn) and argspec.args and argspec.args[0] == "self"
        )
        
        return {
            "name": getattr(callable_fn, "__name__", "<unknown>"),
            "parameters": parameters,
            "required_parameters": required_params,
            "optional_parameters": optional_params,
            "has_varargs": argspec.varargs is not None,
            "has_varkw": argspec.varkw is not None,
            "has_return_annotation": return_annotation is not None,
            "return_annotation": str(return_annotation) if return_annotation else None,
            "is_method": is_method,
        }
    except Exception as e:
        raise ValueError(f"Error analyzing callable signature: {e}") from e


class Tool(Runnable):
    """A Tool is a Runnable that wraps a callable function or method.

    Tools automatically introspect the handler function to:
    - Generate parameter models from function signatures
    - Determine execution characteristics (sync/async, generator, etc.)
    - Extract return type annotations
    - Auto-generate names from function names

    Tools are the basic building blocks that can be used standalone or
    composed into more complex Agents and Workflows.
    """

    handler: SkipValidation[Callable[..., Any]]
    """The callable function or method that this tool wraps."""

    @model_validator(mode="before")
    @classmethod
    def validate_handler_and_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate handler and auto-generate tool name if not provided.

        This validator runs before Pydantic model creation and:
        1. Ensures the handler is a proper function, not a Runnable instance
        2. Automatically extracts the function name to use as the tool name when no explicit name is provided

        Args:
            values: Raw input values for tool creation

        Returns:
            Updated values dict with name field populated

        Raises:
            ValueError: If handler is missing, is a Runnable, has no __name__, or is a lambda without explicit name
        """
        handler = values.get("handler", None)
        if handler is None:
            raise ValueError("You must provide a handler when creating a tool.")

        # Check if handler is a Runnable instance
        if isinstance(handler, Runnable):
            raise ValueError(
                "Handler cannot be a Runnable instance. Tools should wrap any other python callable. "
                "If you want to compose Runnables, use an Agent or Workflow instead, or modify the properties of the Runnable itself."
            )

        if "name" not in values:
            handler_name = getattr(handler, "__name__", None)
            if handler_name is None:
                raise ValueError(
                    "Handler must be a function or lambda with a __name__ attribute. "
                    "If you are using a callable object or functools.partial, please provide a 'name' explicitly."
                )
            if handler_name == "<lambda>":
                raise ValueError("A name must be specified when using a lambda function as a tool.")
            values["name"] = handler_name

        return values

    def model_post_init(self, __context: Any) -> None:
        """Initialize tool-specific attributes after Pydantic model creation.

        This method introspects the handler function to determine its execution
        characteristics, which are used by the base Runnable class to determine
        how to execute the handler.
        """
        super().model_post_init(__context)
        self._path = self.name

        inspect_result = self._inspect_callable(
            self.handler,
            allow_required_params=True,
            allow_gen=True,
            allow_async_gen=True,
        )

        self._is_orchestrator = False
        self._is_coroutine = inspect_result["is_coroutine"]
        self._is_gen = inspect_result["is_gen"]
        self._is_async_gen = inspect_result["is_async_gen"]
        self._dependencies = inspect_result["dependencies"]

    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"

    @override
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """See base class."""
        params_model_name = self.name.title().replace("_", "") + "Params"
        params_model = create_model_from_handler(name=params_model_name, handler=self.handler)
        return params_model

    @override
    @computed_field
    @cached_property
    def return_model(self) -> Any:
        """See base class."""
        handler_argspec = inspect.getfullargspec(self.handler)
        handler_return_annotation = handler_argspec.annotations.get("return", Any)
        return handler_return_annotation

    def get_handler_info(self) -> dict[str, Any]:
        """
        Get detailed information about the tool's handler function.
        
        Returns:
            Dictionary containing:
            - name: Handler function name
            - is_coroutine: Whether handler is async
            - is_generator: Whether handler is a generator
            - is_async_generator: Whether handler is an async generator
            - parameters: List of parameter names
            - required_parameters: List of required parameter names
            - has_return_annotation: Whether return type is annotated
            - return_annotation: Return type annotation if present
        """
        inspect_result = self._inspect_callable(
            self.handler,
            allow_required_params=True,
            allow_gen=True,
            allow_async_gen=True,
        )
        
        handler_argspec = inspect.getfullargspec(self.handler)
        parameters = handler_argspec.args
        required_params = parameters[: len(parameters) - len(handler_argspec.defaults or [])]
        return_annotation = handler_argspec.annotations.get("return", None)
        
        return {
            "name": self.handler.__name__,
            "is_coroutine": inspect_result["is_coroutine"],
            "is_generator": inspect_result["is_gen"],
            "is_async_generator": inspect_result["is_async_gen"],
            "parameters": parameters,
            "required_parameters": required_params,
            "has_return_annotation": return_annotation is not None,
            "return_annotation": str(return_annotation) if return_annotation else None,
        }

    def validate_handler_signature(self) -> dict[str, Any]:
        """
        Validate the handler function signature and return validation results.
        
        Returns:
            Dictionary containing:
            - is_valid: Whether signature is valid
            - errors: List of error messages
            - warnings: List of warning messages
        """
        errors = []
        warnings = []
        
        if not callable(self.handler):
            errors.append("Handler is not callable")
            return {"is_valid": False, "errors": errors, "warnings": warnings}
        
        try:
            handler_argspec = inspect.getfullargspec(self.handler)
            
            if handler_argspec.varargs is not None:
                warnings.append("Handler uses *args, which may cause parameter validation issues")
            
            if handler_argspec.varkw is not None:
                warnings.append("Handler uses **kwargs, which may cause parameter validation issues")
            
            if handler_argspec.kwonlyargs and not handler_argspec.kwonlydefaults:
                warnings.append("Handler has keyword-only arguments without defaults")
            
            return_annotation = handler_argspec.annotations.get("return", None)
            if return_annotation is None:
                warnings.append("Handler has no return type annotation")
            
        except Exception as e:
            errors.append(f"Error inspecting handler signature: {e}")
            return {"is_valid": False, "errors": errors, "warnings": warnings}
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def get_tool_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary of the tool's configuration.
        
        Returns:
            Dictionary containing tool summary information.
        """
        handler_info = self.get_handler_info()
        signature_validation = self.validate_handler_signature()
        
        return {
            "name": self.name,
            "description": self.description,
            "path": self._path,
            "handler_info": handler_info,
            "signature_validation": signature_validation,
            "has_command": self.command is not None,
            "command": self.command,
            "metadata": self.metadata,
            "is_orchestrator": self._is_orchestrator,
            "execution_type": self._get_execution_type(),
        }

    def _get_execution_type(self) -> str:
        """Get a human-readable execution type description."""
        if self._is_async_gen:
            return "async_generator"
        elif self._is_gen:
            return "generator"
        elif self._is_coroutine:
            return "async"
        else:
            return "sync"
