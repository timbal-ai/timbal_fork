import asyncio
import importlib
import inspect
import json
import re
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine
from functools import cached_property
from pathlib import Path
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    ValidationError,
    computed_field,
    model_validator,
)
from uuid_extensions import uuid7

from ..errors import InterruptError, bail
from ..state import get_run_context
from ..types.content import CustomContent, TextContent, ToolUseContent
from ..types.events import BaseEvent, OutputEvent
from ..types.message import Message
from ..utils import coerce_to_dict, dump
from .llm_router import Model, _llm_router
from .runnable import Runnable, RunnableLike
from .skill import ReadSkill, Skill
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.agent")

# Regex patterns for system prompt function resolution
SYSTEM_PROMPT_FN_PATTERN = re.compile(r"\{[a-zA-Z0-9_]*::[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)*\}")
MODIFIED_SYSTEM_PROMPT_FN_PATTERN = re.compile(r"\{[a-zA-Z0-9_]*::[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)*::modified\}")


def extract_system_prompt_patterns(system_prompt: str) -> list[dict[str, Any]]:
    """
    Extract and validate all system prompt function patterns from a string.
    
    Args:
        system_prompt: The system prompt string to analyze
        
    Returns:
        List of dictionaries containing pattern information:
        - 'pattern': The full matched pattern (e.g., '{module::function}')
        - 'path': The path without braces (e.g., 'module::function')
        - 'parts': List of path parts (e.g., ['module', 'function'])
        - 'is_modified': Whether this is a modified pattern
        - 'start': Start position in the string
        - 'end': End position in the string
    """
    if not system_prompt or not isinstance(system_prompt, str):
        return []
    
    patterns = []
    for match in SYSTEM_PROMPT_FN_PATTERN.finditer(system_prompt):
        text = match.group()
        path = text[1:-1]  # Remove { and }
        path_parts = path.split("::")
        
        patterns.append({
            "pattern": text,
            "path": path,
            "parts": path_parts,
            "is_modified": False,
            "start": match.start(),
            "end": match.end(),
        })
    
    for match in MODIFIED_SYSTEM_PROMPT_FN_PATTERN.finditer(system_prompt):
        text = match.group()
        path = text[1:-1]  # Remove { and }
        path_parts = path.split("::")
        
        patterns.append({
            "pattern": text,
            "path": path,
            "parts": path_parts,
            "is_modified": True,
            "start": match.start(),
            "end": match.end(),
        })
    
    return patterns


def validate_model_format(model: str) -> dict[str, Any]:
    """
    Validate model format and extract provider information.
    
    Args:
        model: Model identifier string (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-haiku-4-5')
        
    Returns:
        Dictionary containing:
        - is_valid: Whether the model format is valid
        - provider: Model provider (e.g., 'openai', 'anthropic')
        - model_name: Model name without provider
        - error: Error message if invalid
    """
    if not model or not isinstance(model, str):
        return {
            "is_valid": False,
            "provider": None,
            "model_name": None,
            "error": "Model must be a non-empty string",
        }
    
    if "/" not in model:
        return {
            "is_valid": False,
            "provider": None,
            "model_name": None,
            "error": f"Invalid model format: '{model}'. Expected 'provider/model-name'",
        }
    
    try:
        provider, model_name = model.split("/", 1)
        if not provider or not model_name:
            return {
                "is_valid": False,
                "provider": None,
                "model_name": None,
                "error": "Provider and model name cannot be empty",
            }
        
        valid_providers = ["anthropic", "openai"]
        if provider not in valid_providers:
            return {
                "is_valid": True,
                "provider": provider,
                "model_name": model_name,
                "error": None,
                "warning": f"Unknown provider '{provider}'. Known providers: {valid_providers}",
            }
        
        return {
            "is_valid": True,
            "provider": provider,
            "model_name": model_name,
            "error": None,
        }
    except ValueError:
        return {
            "is_valid": False,
            "provider": None,
            "model_name": None,
            "error": f"Invalid model format: '{model}'",
        }


class AgentParams(BaseModel):
    """Input parameters for Agent execution. Use either 'prompt' or 'messages', not both."""

    model_config = ConfigDict(extra="allow")

    prompt: Message | None = Field(
        None,
        description="Single input message. Memory is automatically resolved from previous runs.",
    )
    messages: list[Message] | None = Field(
        None,
        description="Explicit list of messages. No automatic memory resolution.",
    )

    @model_validator(mode="after")
    def validate_prompt_or_messages(self) -> "AgentParams":
        """Ensure exactly one of prompt or messages is set."""
        if self.prompt is not None and self.messages is not None:
            logger.warning("Calling agent with both 'prompt' and 'messages'. Using 'messages'.")
        if self.prompt is None and self.messages is None:
            raise ValueError("Must specify either 'prompt' or 'messages'.")
        return self

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:
        """Custom schema for using agents as tools."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "object",
                    "title": "TimbalMessage",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["user"],
                        },
                        "content": {
                            "type": "array",
                            "items": {},
                        },
                    },
                }
            },
            "required": ["prompt"],
        }


class Agent(Runnable):
    """Orchestrates LLM interactions with autonomous tool calling."""

    model: Model | str
    """LLM model identifier (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-haiku-4-5')."""
    system_prompt: str | Callable[[], str] | Callable[[], Coroutine[Any, Any, str]] | None = None
    """System prompt. Can be a string, sync callable, or async callable returning a string."""
    tools: list[SkipValidation[RunnableLike]] = []
    """List of tools available to the agent. Can be functions, dicts, or Runnable objects."""
    skills_path: str | Path | None = None
    """Path to the skills directory."""
    max_iter: int = 10
    """Maximum number of LLM->tool call iterations before stopping."""
    model_params: dict[str, Any] = {}
    """Model parameters to pass to the agent."""
    output_model: type[BaseModel] | None = None
    """BaseModel to generate a structured output."""

    _llm: Tool = PrivateAttr()
    """Internal LLM tool instance."""
    _system_prompt_templates: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Template patterns {module::func} mapped to their callables."""
    _system_prompt_fn: Callable | None = PrivateAttr(default=None)
    """Callable passed as system_prompt."""
    _system_prompt_fn_is_async: bool = PrivateAttr(default=False)
    """Whether _system_prompt_fn is async."""

    def model_post_init(self, __context: Any) -> None:
        """Initialize agent after Pydantic model creation."""
        super().model_post_init(__context)
        self._path = self.name

        # Handle callable system_prompt
        if callable(self.system_prompt):
            self._system_prompt_fn = self.system_prompt
            self._system_prompt_fn_is_async = self._inspect_callable(self.system_prompt)["is_coroutine"]
            self.system_prompt = None

        if self.system_prompt:
            for match in SYSTEM_PROMPT_FN_PATTERN.finditer(self.system_prompt):
                text = match.group()
                path = text[1:-1]  # Remove { and }
                path_parts = path.split("::")
                assert len(path_parts) >= 2, (
                    f"Invalid path format for system prompt: {path}. Review the SYSTEM_PROMPT_FN_PATTERN regex."
                )
                module, fn_i = None, None
                if path_parts[0] == "":
                    # File-relative import: look in caller's globals first
                    frame = inspect.currentframe()
                    caller_globals = None
                    while frame:
                        frame = frame.f_back
                        if frame is None:
                            break
                        frame_self = frame.f_locals.get("self")
                        if not isinstance(frame_self, Agent):
                            caller_globals = frame.f_globals
                            break
                    assert caller_globals is not None, "Could not determine caller globals for Agent constructor."

                    # Try to resolve from caller's globals directly (handles same-file definitions)
                    fn = caller_globals
                    try:
                        for attr_name in path_parts[1:]:
                            fn = fn[attr_name] if isinstance(fn, dict) else getattr(fn, attr_name)
                        logger.info(f"Resolved callable '{text}' from caller's globals")
                    except (KeyError, AttributeError):
                        # If not in globals, try loading the module
                        caller_file = Path(caller_globals.get("__file__", "")).expanduser().resolve()
                        agent_path = caller_file
                        for fn_i in range(1, len(path_parts)):
                            module_path = agent_path / "/".join(path_parts[:-fn_i])
                            try:
                                # Use absolute path as module identifier
                                module_path_str = str(module_path.resolve())
                                # Check if module is already loaded in sys.modules by absolute path
                                if module_path_str in sys.modules:
                                    module = sys.modules[module_path_str]
                                    logger.info(
                                        f"Using already loaded module '{module_path}' for system prompt callable '{text}'"
                                    )
                                    break
                                # Load the module if not already loaded
                                module_spec = importlib.util.spec_from_file_location(
                                    module_path_str, module_path.as_posix()
                                )
                                if not module_spec or not module_spec.loader:
                                    raise ValueError(f"Failed to load module {module_path}")
                                module = importlib.util.module_from_spec(module_spec)
                                # Register in sys.modules BEFORE executing to prevent re-entry
                                sys.modules[module_path_str] = module
                                module_spec.loader.exec_module(module)
                                logger.info(f"Loaded module '{module_path}' for system prompt callable '{text}'")
                                break
                            except Exception:
                                pass
                        else:
                            for fn_i in range(1, len(path_parts)):
                                module_path = ".".join(path_parts[:-fn_i])
                                try:
                                    module = importlib.import_module(module_path)
                                    logger.info(f"Loaded module '{module_path}' for system prompt callable '{text}'")
                                    break
                                except Exception:
                                    pass
                        fn = module
                        for j in path_parts[-fn_i:]:
                            fn = getattr(fn, j)

                    inspect_result = self._inspect_callable(fn)
                    self._system_prompt_templates[text] = {
                        "start": match.start(),
                        "end": match.end(),
                        "callable": fn,
                        "is_coroutine": inspect_result["is_coroutine"],
                    }
                else:
                    # Package import (e.g., {os::getcwd})
                    for fn_i in range(1, len(path_parts)):
                        module_path = ".".join(path_parts[:-fn_i])
                        try:
                            module = importlib.import_module(module_path)
                            logger.info(f"Loaded module '{module_path}' for system prompt callable '{text}'")
                            break
                        except Exception:
                            pass
                    fn = module
                    for j in path_parts[-fn_i:]:
                        fn = getattr(fn, j)
                    inspect_result = self._inspect_callable(fn)
                    self._system_prompt_templates[text] = {
                        "start": match.start(),
                        "end": match.end(),
                        "callable": fn,
                        "is_coroutine": inspect_result["is_coroutine"],
                    }

        model_provider, model_name = self.model.split("/", 1)
        if model_provider == "anthropic":
            if not self.model_params.get("max_tokens"):
                raise ValueError("'max_tokens' is required for claude models.")

        self._llm = Tool(
            name="llm",
            handler=_llm_router,
            default_params=self.model_params,
            metadata={
                "type": "LLM",
                "model_provider": model_provider,
                "model_name": model_name,
            },
        )
        self._llm.nest(self._path)

        if self.skills_path is not None:
            self.skills_path = Path(self.skills_path).expanduser().resolve()
            if not self.skills_path.exists() or not self.skills_path.is_dir():
                raise ValueError(
                    f"Skills directory {self.skills_path} does not exist or is not a directory. Skipping..."
                )
            for skill_path in self.skills_path.iterdir():
                skill = Skill(path=skill_path)
                self.tools.append(skill)

        if self.output_model:
            output_model_tool = Tool(
                name="output_model_tool",
                description="Use it always before providing the final answer to give the structured output.",
                handler=lambda x: x,
            )
            output_model_tool.params_model = self.output_model
            self.tools.append(output_model_tool)

        # Normalize tools and prevent duplicate names
        names = set()
        skills_metadata = []
        for i, tool in enumerate(self.tools):
            # ToolSet resolved later in _resolve_tools()
            if isinstance(tool, ToolSet):
                if isinstance(tool, Skill):
                    if tool.name in names:
                        raise ValueError(f"Skill '{tool.name}' already exists. You can only add a skill once.")
                    names.add(tool.name)
                    skills_metadata.append(f"- **{tool.name}**: {tool.description}")
                continue
            if not isinstance(tool, Runnable):
                if isinstance(tool, dict):
                    tool = Tool(**tool)
                else:
                    tool = Tool(handler=tool)
            if tool.name in names:
                raise ValueError(f"Tool {tool.name} already exists. You can only add a tool once.")
            names.add(tool.name)
            tool.nest(self._path)
            self.tools[i] = tool

        if skills_metadata:
            read_skill_tool = ReadSkill()
            read_skill_tool.nest(self._path)
            self.tools.append(read_skill_tool)
            if not isinstance(self.system_prompt, str):
                self.system_prompt = ""
            self.system_prompt += f"""
<skills>
Skills provide additional knowledge of a specific topic. The following skills are available:
{"\n".join(skills_metadata)}
In skills documentation, you will encounter references to additional files.
If the file is relevant for the user query, USE the `read_skill` tool to get its content.
</skills>"""

        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True

    def validate_configuration(self) -> dict[str, Any]:
        """
        Validate agent configuration and return validation results.
        
        Returns:
            Dictionary containing validation results:
            - is_valid: Whether configuration is valid
            - errors: List of error messages
            - warnings: List of warning messages
            - model_info: Information about the model configuration
        """
        errors = []
        warnings = []
        
        # Validate model
        if not self.model:
            errors.append("Model is required")
        else:
            try:
                model_provider, model_name = self.model.split("/", 1)
                if model_provider not in ["anthropic", "openai"]:
                    warnings.append(f"Unknown model provider: {model_provider}")
            except ValueError:
                errors.append(f"Invalid model format: {self.model}. Expected 'provider/model-name'")
        
        # Validate max_iter
        if self.max_iter < 1:
            errors.append("max_iter must be at least 1")
        elif self.max_iter > 100:
            warnings.append(f"max_iter is very high ({self.max_iter}), this may cause long execution times")
        
        # Validate skills_path if provided
        if self.skills_path is not None:
            skills_path = Path(self.skills_path).expanduser().resolve()
            if not skills_path.exists():
                errors.append(f"Skills path does not exist: {self.skills_path}")
            elif not skills_path.is_dir():
                errors.append(f"Skills path is not a directory: {self.skills_path}")
        
        # Validate system prompt patterns
        if self.system_prompt and isinstance(self.system_prompt, str):
            try:
                patterns = extract_system_prompt_patterns(self.system_prompt)
                if patterns:
                    logger.debug(f"Found {len(patterns)} system prompt patterns")
            except Exception as e:
                warnings.append(f"Error parsing system prompt patterns: {e}")
        
        # Validate tools
        tool_names = self.get_tool_names()
        if len(tool_names) != len(set(tool_names)):
            duplicates = [name for name in tool_names if tool_names.count(name) > 1]
            errors.append(f"Duplicate tool names found: {set(duplicates)}")
        
        model_info = {}
        if self.model:
            try:
                model_provider, model_name = self.model.split("/", 1)
                model_info = {
                    "provider": model_provider,
                    "name": model_name,
                    "full_model": self.model,
                }
            except ValueError:
                pass
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "model_info": model_info,
        }

    def get_iteration_status(self, current_iteration: int) -> dict[str, Any]:
        """
        Get status information about the current iteration.
        
        Args:
            current_iteration: Current iteration number (0-indexed)
            
        Returns:
            Dictionary containing:
            - current_iteration: Current iteration number
            - max_iterations: Maximum allowed iterations
            - remaining_iterations: Number of iterations remaining
            - progress_percentage: Progress as percentage
            - is_max_reached: Whether max iterations have been reached
        """
        remaining = max(0, self.max_iter - current_iteration)
        progress = min(100, (current_iteration / self.max_iter * 100)) if self.max_iter > 0 else 0
        
        return {
            "current_iteration": current_iteration,
            "max_iterations": self.max_iter,
            "remaining_iterations": remaining,
            "progress_percentage": round(progress, 2),
            "is_max_reached": current_iteration >= self.max_iter,
        }

    def get_agent_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary of the agent's configuration and state.
        
        Returns:
            Dictionary containing agent summary information.
        """
        tool_stats = self.get_tool_statistics()
        config_validation = self.validate_configuration()
        
        summary = {
            "name": self.name,
            "model": self.model,
            "max_iter": self.max_iter,
            "has_system_prompt": self.system_prompt is not None,
            "has_skills_path": self.skills_path is not None,
            "has_output_model": self.output_model is not None,
            "tool_statistics": tool_stats,
            "configuration_validation": config_validation,
        }
        
        if self.system_prompt and isinstance(self.system_prompt, str):
            patterns = extract_system_prompt_patterns(self.system_prompt)
            summary["system_prompt_patterns_count"] = len(patterns)
        
        return summary

    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
        self._llm.nest(self._path)
        for tool in self.tools:
            tool.nest(self._path)

    @override
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """See base class."""
        return AgentParams

    @override
    @computed_field
    @cached_property
    def return_model(self) -> Any:
        """See base class."""
        return Message

    async def _resolve_system_prompt(self) -> str | None:
        """Resolve system prompt by executing callable or embedded template functions."""
        if self._system_prompt_fn is not None:
            return await self._execute_runtime_callable(self._system_prompt_fn, self._system_prompt_fn_is_async)

        if not self.system_prompt:
            return None
        if not self._system_prompt_templates:
            return self.system_prompt

        # Execute template functions in parallel
        system_prompt_tasks = []
        for _, v in self._system_prompt_templates.items():
            callable_fn = v["callable"]
            system_prompt_tasks.append(self._execute_runtime_callable(callable_fn, v["is_coroutine"]))
        results = await asyncio.gather(*system_prompt_tasks)

        # Substitute results into template
        system_prompt = self.system_prompt
        for (k, _), result in zip(self._system_prompt_templates.items(), results, strict=False):
            system_prompt = system_prompt.replace(k, str(result) if result is not None else "")

        return system_prompt

    def validate_system_prompt_patterns(self) -> list[dict[str, Any]]:
        """
        Validate and extract all system prompt function patterns.
        
        Returns:
            List of pattern dictionaries with validation information.
            Raises ValueError if any patterns are invalid.
        """
        if not self.system_prompt or not isinstance(self.system_prompt, str):
            return []
        
        patterns = extract_system_prompt_patterns(self.system_prompt)
        
        for pattern_info in patterns:
            path_parts = pattern_info["parts"]
            if len(path_parts) < 2:
                raise ValueError(
                    f"Invalid path format for system prompt pattern '{pattern_info['pattern']}': "
                    f"expected at least 2 parts separated by '::', got {len(path_parts)}"
                )
        
        if patterns:
            logger.debug(
                f"Found {len(patterns)} system prompt pattern(s)",
                patterns=[p["pattern"] for p in patterns],
            )
        
        return patterns

    def get_tool_names(self) -> list[str]:
        """
        Get a list of all available tool names.
        
        Returns:
            List of tool names (strings) available to this agent.
        """
        names = []
        for tool in self.tools:
            if isinstance(tool, ToolSet):
                continue
            if isinstance(tool, Runnable):
                names.append(tool.name)
            elif isinstance(tool, dict):
                names.append(tool.get("name", "unknown"))
            else:
                names.append(getattr(tool, "__name__", "unknown"))
        return names

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool with the given name exists.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool exists, False otherwise
        """
        return tool_name in self.get_tool_names()

    def get_tool_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the agent's tools configuration.
        
        Returns:
            Dictionary containing:
            - total_tools: Total number of tools
            - tool_names: List of all tool names
            - toolsets_count: Number of ToolSet instances
            - skills_count: Number of Skill instances
            - commands_count: Number of tools with commands
        """
        tool_names = []
        toolsets_count = 0
        skills_count = 0
        commands_count = 0
        
        for tool in self.tools:
            if isinstance(tool, ToolSet):
                toolsets_count += 1
                if isinstance(tool, Skill):
                    skills_count += 1
                continue
            
            if isinstance(tool, Runnable):
                tool_names.append(tool.name)
                if hasattr(tool, "command") and tool.command:
                    commands_count += 1
            elif isinstance(tool, dict):
                name = tool.get("name", "unknown")
                tool_names.append(name)
                if tool.get("command"):
                    commands_count += 1
        
        return {
            "total_tools": len(tool_names),
            "tool_names": tool_names,
            "toolsets_count": toolsets_count,
            "skills_count": skills_count,
            "commands_count": commands_count,
        }

    async def resolve_memory(self) -> None:
        """Resolve conversation memory from previous agent trace."""
        run_context = get_run_context()
        assert run_context is not None, "Run context not found"

        current_span = run_context.current_span()
        if current_span.memory:
            return
        current_span.memory = [Message.validate(current_span.input.get("prompt", ""))]

        # Subagents have isolated context
        if current_span.parent_call_id is not None:
            return

        # User can override the message list
        input_messages = current_span.input.get("messages", [])
        if input_messages:
            current_span.memory = [Message.validate(m) for m in input_messages]
            return

        if not run_context.parent_id:
            return
        parent_trace = await run_context._get_parent_trace()
        if parent_trace is None:
            logger.error(
                "Parent trace not found. Continuing without memory...",
                parent_id=run_context.parent_id,
                run_id=run_context.id,
            )
            return

        self_spans = parent_trace.get_path(self._path)
        if not len(self_spans):
            return
        if len(self_spans) > 1:
            # TODO Handle multiple call_ids for this agent (we could access step spans)
            raise NotImplementedError("Multiple spans for the same agent are not supported yet.")
        previous_span = self_spans[0]
        if hasattr(previous_span, "in_context_skills"):
            current_span.in_context_skills = previous_span.in_context_skills

        # >= 1.1.0: memory stored in agent span
        if isinstance(previous_span.memory, list):
            memory = [Message.validate(m) for m in previous_span.memory]
        else:
            # < 1.1.0: extract from LLM calls
            llm_spans = parent_trace.get_path(self._llm._path)
            if not len(llm_spans):
                return
            llm_input_messages = llm_spans[-1].input.get("messages", [])
            llm_output_message = llm_spans[-1].output
            memory = [*[Message.validate(m) for m in llm_input_messages], Message.validate(llm_output_message)]

        # Ensure interrupted tool calls have corresponding results.
        # When resuming from memory, the last assistant message may contain tool_use blocks
        # that were interrupted before their results could be recorded. The LLM expects every
        # tool_use to have a corresponding tool_result, so we need to synthesize error results
        # for any missing ones.
        #
        # We iterate in reverse to detect if there's any non-tool_use content (text, etc.)
        # after a server_tool_use block. If there is, the server tool completed and the LLM
        # already continued, so no synthetic result is needed.
        #
        # For regular tool_use: append a tool_result message with an error.
        # For server_tool_use (e.g., web_search): append an inline error result in the same
        # message, using the provider's expected error format (currently Anthropic only).
        # TODO OpenAI & other server tool use providers
        has_followup_after_server_tool_use = False
        for content in memory[-1].content[::-1]:
            if content.type != "tool_use":
                has_followup_after_server_tool_use = True
                continue
            if content.is_server_tool_use:
                if has_followup_after_server_tool_use:
                    continue
                # Server tool use was interrupted before completion. Synthesize an error result.
                memory[-1].content.append(
                    CustomContent(
                        value={
                            "type": "web_search_tool_result",
                            "tool_use_id": content.id,
                            "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
                        },
                    )
                )
            else:
                # Regular tool use was interrupted. Append a tool_result message with error.
                memory.append(
                    Message.validate(
                        {
                            "role": "tool",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "id": content.id,
                                    "content": "ERROR: There was an unexpected error executing the tool.",
                                }
                            ],
                        }
                    )
                )
        current_span.memory = memory + current_span.memory

    def get_memory_statistics(self) -> dict[str, Any] | None:
        """
        Get statistics about the current agent memory.
        
        Returns:
            Dictionary containing memory statistics:
            - message_count: Total number of messages
            - user_messages: Number of user messages
            - assistant_messages: Number of assistant messages
            - tool_messages: Number of tool result messages
            - has_memory: Whether memory exists
            None if run context is not available.
        """
        run_context = get_run_context()
        if run_context is None:
            return None
        
        current_span = run_context.current_span()
        if not current_span.memory:
            return {
                "has_memory": False,
                "message_count": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "tool_messages": 0,
            }
        
        user_count = sum(1 for msg in current_span.memory if msg.role == "user")
        assistant_count = sum(1 for msg in current_span.memory if msg.role == "assistant")
        tool_count = sum(1 for msg in current_span.memory if msg.role == "tool")
        
        return {
            "has_memory": True,
            "message_count": len(current_span.memory),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "tool_messages": tool_count,
        }

    async def _resolve_tools(self, i: int) -> tuple[list[Tool], dict[str, Tool]]:
        """Resolve the tools to be provided to the LLM."""
        if i >= self.max_iter:
            return [], {}
        tools = []
        tools_names = set()
        commands = {}
        for t in self.tools:
            if isinstance(t, ToolSet):
                resolved_toolset = await t.resolve()
                for tool in resolved_toolset:
                    if tool.name in tools_names:
                        logger.warning(f"Tool with name '{tool.name}' already exists. You can only add a tool once.")
                    else:
                        tool.nest(self._path)
                        tools.append(tool)
                        tools_names.add(tool.name)
                        if tool.command:
                            commands[tool.command] = tool
            else:
                if t.name in tools_names:
                    logger.warning(f"Tool with name '{t.name}' already exists. You can only add a tool once.")
                else:
                    tools.append(t)
                    tools_names.add(t.name)
                    if t.command:
                        commands[t.command] = t

        if self._bg_tasks:
            get_background_task_tool = Tool(
                name="get_background_task",
                description="Get the status and events of a background task.",
                handler=self.get_background_task,
            )
            get_background_task_tool.nest(self._path)
            tools.append(get_background_task_tool)
            tools_names.add("get_background_task")
            # Add to commands dict if the tool has a command attribute
            # if get_background_task_tool.command:
            #     commands[get_background_task_tool.command] = get_background_task_tool

        return tools, commands

    async def _multiplex_tools(self, tools: list[Tool], tool_calls: list[ToolUseContent]) -> AsyncGenerator[Any, None]:
        """Execute multiple tool calls concurrently and multiplex their events."""
        queue = asyncio.Queue()
        tasks = []

        async def consume_tool(tool_call: ToolUseContent):
            tool = next((t for t in tools if t.name == tool_call.name), None)
            assert tool is not None, f"Tool {tool_call.name} not found"
            try:
                async for event in tool(**tool_call.input):
                    # Link tool call id to span for memory resolution
                    if event.type == "START":
                        tool_call_id = event.call_id
                        tool_call_span = get_run_context()._trace[tool_call_id]
                        tool_call_span.metadata["tool_call_id"] = tool_call.id
                    await queue.put((tool_call, event))
            finally:
                await queue.put((tool_call, None))

        try:
            for tc in tool_calls:
                task = asyncio.create_task(consume_tool(tc))
                tasks.append(task)

            remaining = len(tool_calls)
            while remaining > 0:
                tool_call, event = await queue.get()
                if event is None:
                    remaining -= 1
                else:
                    yield tool_call, event
        except (asyncio.CancelledError, GeneratorExit, InterruptError):
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def analyze_tool_calls_from_memory(self) -> dict[str, Any]:
        """
        Analyze tool calls from the current memory state.
        
        Returns:
            Dictionary containing:
            - total_tool_calls: Total number of tool calls
            - tool_call_names: List of tool names that were called
            - tool_call_counts: Dictionary mapping tool names to call counts
            - pending_tool_calls: List of tool calls without results
        """
        run_context = get_run_context()
        if run_context is None:
            return {
                "total_tool_calls": 0,
                "tool_call_names": [],
                "tool_call_counts": {},
                "pending_tool_calls": [],
            }
        
        current_span = run_context.current_span()
        if not current_span.memory:
            return {
                "total_tool_calls": 0,
                "tool_call_names": [],
                "tool_call_counts": {},
                "pending_tool_calls": [],
            }
        
        tool_call_ids = set()
        tool_call_counts = {}
        tool_call_names = []
        pending_tool_calls = []
        
        for message in current_span.memory:
            if message.role == "assistant":
                for content in message.content:
                    if isinstance(content, ToolUseContent):
                        tool_call_ids.add(content.id)
                        tool_name = content.name
                        tool_call_names.append(tool_name)
                        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                        pending_tool_calls.append({
                            "id": content.id,
                            "name": tool_name,
                            "input": content.input,
                        })
            elif message.role == "tool":
                for content in message.content:
                    if isinstance(content, dict) and content.get("type") == "tool_result":
                        tool_call_id = content.get("id")
                        if tool_call_id in tool_call_ids:
                            tool_call_ids.remove(tool_call_id)
                            pending_tool_calls = [tc for tc in pending_tool_calls if tc["id"] != tool_call_id]
        
        return {
            "total_tool_calls": len(tool_call_names),
            "tool_call_names": list(set(tool_call_names)),
            "tool_call_counts": tool_call_counts,
            "pending_tool_calls": pending_tool_calls,
        }

    def get_execution_state(self) -> dict[str, Any]:
        """
        Get the current execution state of the agent.
        
        Returns:
            Dictionary containing execution state information.
        """
        run_context = get_run_context()
        if run_context is None:
            return {
                "has_run_context": False,
                "is_running": False,
            }
        
        current_span = run_context.current_span()
        memory_stats = self.get_memory_statistics()
        tool_analysis = self.analyze_tool_calls_from_memory()
        
        return {
            "has_run_context": True,
            "is_running": current_span is not None,
            "run_id": run_context.id if run_context else None,
            "parent_id": run_context.parent_id if run_context else None,
            "memory_statistics": memory_stats,
            "tool_analysis": tool_analysis,
            "agent_path": self._path,
        }

    def get_tool_by_name(self, tool_name: str) -> Tool | None:
        """
        Get a tool instance by its name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found, None otherwise
        """
        for tool in self.tools:
            if isinstance(tool, ToolSet):
                continue
            if isinstance(tool, Runnable) and tool.name == tool_name:
                return tool
            elif isinstance(tool, dict) and tool.get("name") == tool_name:
                return None
        return None

    def get_tools_by_type(self, tool_type: type) -> list[Tool]:
        """
        Get all tools that are instances of a specific type.
        
        Args:
            tool_type: Type to filter by (e.g., Skill, Tool)
            
        Returns:
            List of tools matching the type
        """
        tools = []
        for tool in self.tools:
            if isinstance(tool, tool_type):
                tools.append(tool)
        return tools

    def get_commands_mapping(self) -> dict[str, Tool]:
        """
        Get a mapping of command strings to tool instances.
        
        Returns:
            Dictionary mapping command strings to Tool instances
        """
        commands = {}
        for tool in self.tools:
            if isinstance(tool, ToolSet):
                continue
            if isinstance(tool, Runnable) and tool.command:
                commands[tool.command] = tool
        return commands

    def validate_tool_configurations(self) -> dict[str, Any]:
        """
        Validate all tool configurations in the agent.
        
        Returns:
            Dictionary containing:
            - is_valid: Whether all tools are valid
            - tool_validations: List of validation results for each tool
            - errors: List of global errors
            - warnings: List of global warnings
        """
        errors = []
        warnings = []
        tool_validations = []
        
        for tool in self.tools:
            if isinstance(tool, ToolSet):
                continue
            
            if isinstance(tool, Runnable):
                if hasattr(tool, "validate_handler_signature"):
                    validation = tool.validate_handler_signature()
                    tool_validations.append({
                        "tool_name": tool.name,
                        "validation": validation,
                    })
                    if not validation["is_valid"]:
                        errors.extend([f"{tool.name}: {e}" for e in validation["errors"]])
                    warnings.extend([f"{tool.name}: {w}" for w in validation["warnings"]])
        
        # Check for duplicate commands
        commands = self.get_commands_mapping()
        if len(commands) != len(set(commands.keys())):
            duplicate_commands = [cmd for cmd in commands.keys() if list(commands.keys()).count(cmd) > 1]
            errors.append(f"Duplicate commands found: {set(duplicate_commands)}")
        
        return {
            "is_valid": len(errors) == 0,
            "tool_validations": tool_validations,
            "errors": errors,
            "warnings": warnings,
        }

    def get_model_requirements(self) -> dict[str, Any]:
        """
        Get model-specific requirements and recommendations.
        
        Returns:
            Dictionary containing:
            - provider: Model provider
            - requires_max_tokens: Whether max_tokens is required
            - recommended_max_tokens: Recommended max_tokens value if applicable
            - supports_tools: Whether the model supports tool calling
            - supports_streaming: Whether the model supports streaming
        """
        if not self.model:
            return {
                "provider": None,
                "requires_max_tokens": False,
                "recommended_max_tokens": None,
                "supports_tools": False,
                "supports_streaming": False,
            }
        
        model_validation = validate_model_format(self.model)
        if not model_validation["is_valid"]:
            return {
                "provider": None,
                "requires_max_tokens": False,
                "recommended_max_tokens": None,
                "supports_tools": False,
                "supports_streaming": False,
            }
        
        provider = model_validation["provider"]
        requires_max_tokens = provider == "anthropic"
        recommended_max_tokens = 4096 if provider == "anthropic" else None
        
        return {
            "provider": provider,
            "requires_max_tokens": requires_max_tokens,
            "recommended_max_tokens": recommended_max_tokens,
            "supports_tools": True,
            "supports_streaming": True,
        }

    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Execute the autonomous agent loop."""
        run_context = get_run_context()
        assert run_context is not None, "Run context is not initialized"
        current_span = run_context.current_span()

        model = kwargs.pop("model", self.model)

        # ? Do we want to allow the user to pass parameterized system prompts
        system_prompt = kwargs.pop("system_prompt", None)
        if not system_prompt:
            system_prompt = await self._resolve_system_prompt()

        await self.resolve_memory()
        # Span memory will also be modified with messages array modification (it's the same object)
        # current_span.memory = messages
        current_span._memory_dump = await dump(
            current_span.memory
        )  # ? Can we optimize the dumping the llm already does next
        # Remove these so the llm runnable doesn't try to use/validate them again
        kwargs.pop("prompt", None)
        kwargs.pop("messages", None)

        async def _process_tool_event(event: BaseEvent, tool_call_id: str, append_to_messages: bool = True):
            """Helper to process tool output events and create tool results."""
            if not isinstance(event, OutputEvent) or event.path.count(".") != self._path.count(".") + 1:
                return
            if event.status.code == "cancelled" and event.status.reason == "early_exit":
                bail(event.status.message)
            content = None
            if event.status.code == "cancelled" and event.status.reason == "early_exit_local":
                msg = event.status.message or "The tool exited early."
                content = f"[Cancelled] {msg}"
            elif event.error is not None:
                content = event.error
            elif isinstance(event.output, Message):
                content = event.output.content
            else:
                content = event.output
            tool_result = Message.validate(
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": tool_call_id,
                            "content": content,
                        }
                    ],
                }
            )
            if append_to_messages:
                current_span.memory.append(tool_result)
            tool_result_dump = await dump(tool_result)
            current_span._memory_dump.append(tool_result_dump)

        i = 0
        while True:
            # ? We could resolve the system prompt at each iteration
            tools, commands = await self._resolve_tools(i)
            if commands:
                # Commands will only be user messages with a single text content
                if len(current_span.memory[-1].content) == 1:
                    content = current_span.memory[-1].content[0]
                    if isinstance(content, TextContent) and content.text.startswith("/"):
                        import shlex

                        args = shlex.split(content.text)
                        command = args[0].strip("/")
                        args = args[1:]
                        # If no command is found, we'll simply let the message pass through the LLM
                        if command in commands:
                            tool = commands[command]
                            tool_input = {}
                            for i, field_name in enumerate(tool.params_model.model_fields.keys()):
                                # Params model preserves the ordering of the fields as they appear in the signature
                                # We grab as many arguments as we can. If there are too few arguments, we'll let the tool params model validator give a better error
                                if i >= len(args):
                                    break
                                tool_input[field_name] = args[i]
                            # Craft a fake tool_use so we can keep this interaction in the agent memory
                            tool_use_id = uuid7(as_type="str").replace("-", "")
                            current_span._memory_dump.append(
                                {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "tool_use", "id": tool_use_id, "name": tool.name, "input": tool_input}
                                    ],
                                }
                            )
                            # Run the tool
                            async for event in tool(**tool_input):
                                await _process_tool_event(event, tool_use_id, append_to_messages=False)
                                if isinstance(event, OutputEvent) and event.output is not None:
                                    current_span.memory.append(
                                        Message.validate(
                                            {
                                                "role": "assistant",
                                                "content": [TextContent(text=str(event.output))],
                                            }
                                        )
                                    )
                                yield event
                            return

            is_output_model = False
            async for event in self._llm(
                model=model,
                messages=current_span.memory,
                system_prompt=system_prompt,
                tools=tools,
                **kwargs,
            ):
                if isinstance(event, OutputEvent):
                    # If the LLM call fails, we want to propagate the error upwards
                    if event.error is not None:
                        raise RuntimeError(event.error)
                    # TODO Test what happens when the LLM is in the middle of thinking, tool use or other than text generation
                    assert isinstance(event.output, Message), (
                        f"Expected event.output to be a Message, got {type(event.output)}"
                    )
                    interrupted = event.status.code == "cancelled" and event.status.reason == "interrupted"
                    # # If the response was interrupted amid
                    # if interrupted:
                    #     for content in event.output.content[::-1]:

                    # Add LLM response to conversation for next iteration
                    current_span.memory.append(event.output)
                    current_span._memory_dump.append(event._output_dump)

                    if self.output_model is not None:
                        for content in event.output.content:
                            if isinstance(content, TextContent):
                                try:
                                    output = coerce_to_dict(content.text)
                                    validated_output = self.output_model(**output)
                                    event.output = validated_output
                                    is_output_model = True
                                    break
                                except (json.JSONDecodeError, ValueError, ValidationError):
                                    logger.error(f"Failed to parse JSON from LLM output: {content.text}")
                                    continue
                    # Propagate the interruption with the processed output
                    if interrupted:
                        raise InterruptError(event.call_id, output=event.output)
                yield event

            if is_output_model:
                break

            tool_calls = [
                content
                for content in current_span.memory[-1].content
                if isinstance(content, ToolUseContent) and not content.is_server_tool_use
            ]

            if not tool_calls:
                break

            async for tool_call, event in self._multiplex_tools(tools, tool_calls):
                await _process_tool_event(event, tool_call.id, append_to_messages=True)
                yield event
            i += 1
