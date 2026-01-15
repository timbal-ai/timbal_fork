import asyncio
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, create_model

from timbal.types.events.start import StartEvent

from ..errors import InterruptError
from ..types.events.output import OutputEvent
from .runnable import Runnable, RunnableLike
from .tool import Tool

logger = structlog.get_logger("timbal.core.workflow")


class Workflow(Runnable):
    """A Workflow is a Runnable that orchestrates execution of multiple steps in a directed acyclic graph (DAG).

    Workflows implement a step-based execution pattern where:
    1. Steps are added as Runnable components with explicit dependencies
    2. Steps can be linked to form execution chains based on data dependencies
    3. All steps execute concurrently while respecting dependency constraints
    4. Failed steps automatically skip their dependent steps to prevent cascading failures
    5. The workflow completes when all executable steps finish

    Workflows support:
    - Automatic step linking based on data key dependencies (e.g., step1.output -> step2.input)
    - Concurrent execution of independent steps for optimal performance
    - DAG validation to prevent circular dependencies
    - Graceful error handling with dependent step skipping
    - Dynamic parameter collection from all constituent steps
    """

    _steps: dict[str, Runnable] = PrivateAttr(default_factory=dict)
    """List of steps to execute in the workflow."""

    def model_post_init(self, __context: Any) -> None:
        """Initialize workflow as an orchestrator with async generator handler."""
        super().model_post_init(__context)
        self._path = self.name

        # Workflows are always orchestrators with async generator handlers
        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True

    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
        # Update paths for internal LLM and all tools
        for step in self._steps.values():
            step.nest(self._path)

    @override
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """See base class."""
        fields = {}
        for step in self._steps.values():
            for param, field_info in step.params_model.__pydantic_fields__.items():
                # If a default is set for the param, we remove this from the model, but allow
                # extra properties to enable overriding these values from kwargs
                if param not in step.default_params:
                    fields[param] = (field_info.annotation, field_info)
        params_model_name = self.name.title().replace("_", "") + "Params"
        return create_model(params_model_name, __config__=ConfigDict(extra="allow"), **fields)

    @override
    @computed_field
    @cached_property
    def return_model(self) -> Any:
        """See base class."""
        # TODO Implement
        return Any

    def get_step_names(self) -> list[str]:
        """Get a list of all step names in the workflow."""
        return list(self._steps.keys())

    def get_step(self, step_name: str) -> Runnable | None:
        """Get a step by name.
        
        Args:
            step_name: Name of the step to retrieve
            
        Returns:
            Runnable step if found, None otherwise
        """
        return self._steps.get(step_name)

    def has_step(self, step_name: str) -> bool:
        """Check if a step exists in the workflow."""
        return step_name in self._steps

    def get_step_dependencies(self, step_name: str) -> dict[str, Any]:
        """Get dependency information for a specific step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary containing:
            - previous_steps: Steps that must complete before this step
            - next_steps: Steps that depend on this step
            - has_conditional: Whether step has a conditional handler
        """
        if step_name not in self._steps:
            raise ValueError(f"Step '{step_name}' not found in workflow.")
        
        step = self._steps[step_name]
        return {
            "previous_steps": list(step.previous_steps),
            "next_steps": list(step.next_steps),
            "has_conditional": step.when is not None,
        }

    def get_workflow_statistics(self) -> dict[str, Any]:
        """Get statistics about the workflow structure.
        
        Returns:
            Dictionary containing:
            - total_steps: Total number of steps
            - step_names: List of all step names
            - total_links: Total number of dependency links
            - is_dag: Whether workflow forms a valid DAG
            - entry_points: Steps with no dependencies
            - exit_points: Steps with no dependents
        """
        total_links = sum(len(step.next_steps) for step in self._steps.values())
        entry_points = [name for name, step in self._steps.items() if not step.previous_steps]
        exit_points = [name for name, step in self._steps.items() if not step.next_steps]
        
        return {
            "total_steps": len(self._steps),
            "step_names": list(self._steps.keys()),
            "total_links": total_links,
            "is_dag": self._is_dag(),
            "entry_points": entry_points,
            "exit_points": exit_points,
        }

    def validate_workflow(self) -> dict[str, Any]:
        """Validate the workflow configuration.
        
        Returns:
            Dictionary containing:
            - is_valid: Whether workflow is valid
            - errors: List of error messages
            - warnings: List of warning messages
        """
        errors = []
        warnings = []
        
        if not self._steps:
            warnings.append("Workflow has no steps")
        
        if not self._is_dag():
            errors.append("Workflow contains cycles and is not a valid DAG")
        
        # Check for orphaned steps (steps that are never reached)
        all_referenced = set()
        for step in self._steps.values():
            all_referenced.update(step.previous_steps)
            all_referenced.update(step.next_steps)
        
        entry_points = [name for name, step in self._steps.items() if not step.previous_steps]
        if not entry_points:
            warnings.append("Workflow has no entry points (all steps have dependencies)")
        
        exit_points = [name for name, step in self._steps.items() if not step.next_steps]
        if not exit_points:
            warnings.append("Workflow has no exit points (all steps have dependents)")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _is_dag(self) -> bool:
        """Check if the workflow forms a valid DAG using depth-first search cycle detection."""
        if not self._steps:
            return True
        
        # States: 0 = unvisited, 1 = visiting, 2 = visited
        state = {step_name: 0 for step_name in self._steps.keys()}

        def _dfs_cycle_detection(step_name: str) -> bool:
            """Depth-first search helper for cycle detection."""
            if state[step_name] == 1:  # Currently visiting - cycle detected
                return False
            if state[step_name] == 2:  # Already visited - no cycle from here
                return True
            
            state[step_name] = 1  # Mark as visiting
            for next_step_name in self._steps[step_name].next_steps:
                if not _dfs_cycle_detection(next_step_name):
                    return False
            state[step_name] = 2  # Mark as visited
            return True

        # Check all components of the graph
        for step_name in self._steps.keys():
            if state[step_name] == 0:
                if not _dfs_cycle_detection(step_name):
                    return False
        return True

    def _validate_step_exists(self, step_name: str, step_type: str = "step") -> None:
        """Validate that a step exists in the workflow."""
        if step_name not in self._steps:
            raise ValueError(f"{step_type.capitalize()} step '{step_name}' not found in workflow.")

    def _link(self, source: str, target: str) -> "Workflow":
        """Internal method to link workflow steps with cycle detection."""
        self._validate_step_exists(source, "source")
        self._validate_step_exists(target, "target")
        
        if source == target:
            raise ValueError(f"Cannot link step '{source}' to itself.")
        
        self._steps[source].next_steps.add(target)
        self._steps[target].previous_steps.add(source)
        
        if not self._is_dag():
            # Revert the link if it creates a cycle
            self._steps[source].next_steps.remove(target)
            self._steps[target].previous_steps.remove(source)
            raise ValueError(f"Linking {source} -> {target} would create a cycle in the workflow.")
        
        return self

    def _normalize_runnable(self, runnable: RunnableLike) -> Runnable:
        """Convert a RunnableLike to a Runnable instance."""
        if isinstance(runnable, Runnable):
            return runnable
        if isinstance(runnable, dict):
            return Tool(**runnable)
        return Tool(handler=runnable)  # type: ignore[call-arg]

    def _collect_dependencies(
        self, runnable: Runnable, depends_on: list[str] | None, when: Callable[[], bool] | None, **kwargs: Any
    ) -> set[str]:
        """Collect all dependencies for a step from various sources."""
        dependencies = set(depends_on or [])
        
        # Collect from runnable's internal dependencies
        dependencies.update(runnable._dependencies)
        dependencies.update(runnable._pre_hook_dependencies)
        dependencies.update(runnable._post_hook_dependencies)
        
        # Collect from conditional handler
        if when:
            inspect_result = runnable._inspect_callable(when)
            runnable.when = {"callable": when, **inspect_result}
            dependencies.update(inspect_result["dependencies"])
        
        # Collect from default params
        runnable._prepare_default_params(kwargs)
        for v in runnable._default_runtime_params.values():
            dependencies.update(v["dependencies"])
        
        return dependencies

    def _initialize_step(self, runnable: Runnable) -> None:
        """Initialize a step's workflow-specific attributes."""
        runnable.nest(self._path)
        runnable.previous_steps = set()
        runnable.next_steps = set()
        runnable.when = None

    # TODO Think how we handle agent model_params vs default_params
    def step(
        self,
        runnable: RunnableLike,
        depends_on: list[str] | None = None,
        when: Callable[[], bool] | None = None,
        **kwargs: Any,
    ) -> "Workflow":
        """Add a step to the workflow with automatic dependency linking.

        Adds a runnable component as a workflow step and automatically creates
        dependency links based on data key analysis. If step parameters reference
        other steps' outputs (e.g., step1.result), those dependencies are
        automatically linked.

        The runnable can be:
        - A Runnable instance
        - A dictionary that will be converted to a Tool
        - A callable that will be wrapped in a Tool

        Args:
            runnable: The runnable component to add as a step
            depends_on: Optional list of steps that must complete before this step
            when: Optional callable that returns a boolean to conditionally execute the step
            **kwargs: Default parameters for the step, also used for dependency analysis

        Returns:
            Self for method chaining
        """
        if depends_on is not None and not isinstance(depends_on, list):
            raise ValueError("depends_on must be a list of step names or None")

        runnable = self._normalize_runnable(runnable)
        
        if runnable.name in self._steps:
            raise ValueError(f"Step '{runnable.name}' already exists in the workflow.")

        self._initialize_step(runnable)
        self._steps[runnable.name] = runnable

        dependencies = self._collect_dependencies(runnable, depends_on, when, **kwargs)
        
        # Validate dependencies exist before linking
        for dep in dependencies:
            self._validate_step_exists(dep, "dependency")
        
        # Create dependency links
        for dep in dependencies:
            logger.info("Linking steps", previous_step=dep, next_step=runnable.name)
            self._link(dep, runnable.name)

        return self

    def _should_skip_dependent(self, next_step: Runnable, completions: dict[str, asyncio.Event], force: bool) -> bool:
        """Determine if a dependent step should be skipped."""
        if force:
            return True
        return all(completions[dep].is_set() for dep in next_step.previous_steps)

    def _skip_next_steps(self, step_name: str, completions: dict[str, asyncio.Event], force: bool = False) -> None:
        """Recursively mark a step and all its dependents as completed (skipped).
        
        Args:
            step_name: The step to mark as completed
            completions: Dictionary mapping step names to completion events
            force: If True (error case), skip all dependents immediately.
                   If False (conditional skip), only skip dependents if ALL their dependencies are completed.
        """
        completions[step_name].set()
        for next_name in self._steps[step_name].next_steps:
            next_step = self._steps[next_name]
            if self._should_skip_dependent(next_step, completions, force):
                self._skip_next_steps(next_name, completions, force=force)

    async def _wait_for_dependencies(self, step: Runnable, completions: dict[str, asyncio.Event]) -> None:
        """Wait for all prerequisite steps to complete."""
        if step.previous_steps:
            await asyncio.gather(*[completions[step_name].wait() for step_name in step.previous_steps])

    def _is_step_already_completed(self, step_name: str, completions: dict[str, asyncio.Event]) -> bool:
        """Check if a step has already been marked as completed."""
        return completions[step_name].is_set()

    async def _process_step_event(
        self, event: Any, step: Runnable, completions: dict[str, asyncio.Event], queue: asyncio.Queue
    ) -> bool:
        """Process a single event from a step execution.
        
        Returns:
            True if step started, False otherwise
        """
        await queue.put(event)
        
        if isinstance(event, StartEvent):
            return True
        
        if isinstance(event, OutputEvent) and event.error is not None:
            logger.info(f"Step {step.name} failed, skipping successors...")
            self._skip_next_steps(step.name, completions, force=True)
        
        return False

    async def _execute_step(
        self, step: Runnable, queue: asyncio.Queue, completions: dict[str, asyncio.Event], **kwargs: Any
    ) -> None:
        """Execute a single step and handle its events."""
        step_started = False
        try:
            async for event in step(**kwargs):
                started = await self._process_step_event(event, step, completions, queue)
                if started:
                    step_started = True
        except Exception as e:
            await queue.put(e)
            return

        if not step_started:
            logger.info(f"Step {step.name} did not start, skipping successors...")
            self._skip_next_steps(step.name, completions)

    async def _enqueue_step_events(
        self, step: Runnable, queue: asyncio.Queue, completions: dict[str, asyncio.Event], **kwargs: Any
    ) -> None:
        """Execute a single workflow step and enqueue its events to the shared queue."""
        await self._wait_for_dependencies(step, completions)
        
        if self._is_step_already_completed(step.name, completions):
            logger.info(f"Skipping {step.name} as it's already marked as completed.")
            await queue.put(None)
            return

        await self._execute_step(step, queue, completions, **kwargs)
        
        completions[step.name].set()
        await queue.put(None)

    def _create_completion_events(self) -> dict[str, asyncio.Event]:
        """Create completion events for all workflow steps."""
        return {step_name: asyncio.Event() for step_name in self._steps.keys()}

    def _create_step_tasks(
        self, queue: asyncio.Queue, completions: dict[str, asyncio.Event], **kwargs: Any
    ) -> list[asyncio.Task]:
        """Create async tasks for all workflow steps."""
        return [
            asyncio.create_task(self._enqueue_step_events(step, queue, completions, **kwargs))
            for step in self._steps.values()
        ]

    def _handle_event(self, event: Any) -> bool:
        """Handle an event from the queue.
        
        Returns:
            True if event should be yielded, False if it's a completion marker
        """
        if isinstance(event, InterruptError):
            raise event
        if isinstance(event, Exception):
            raise event
        return event is not None

    async def _process_event_queue(
        self, queue: asyncio.Queue, num_tasks: int
    ) -> AsyncGenerator[Any, None]:
        """Process events from the queue until all tasks complete."""
        remaining = num_tasks
        while remaining > 0:
            event = await queue.get()
            if self._handle_event(event):
                yield event
            else:
                remaining -= 1

    async def _cleanup_tasks(self, tasks: list[asyncio.Task]) -> None:
        """Cancel and wait for all tasks to complete."""
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Main workflow execution handler implementing concurrent step orchestration.

        This is the core workflow logic that implements concurrent step execution:
        1. Creates completion events for all steps to coordinate dependencies
        2. Launches all steps concurrently as async tasks
        3. Each step waits for its prerequisites before executing
        4. Multiplexes events from all steps as they become available
        5. Continues until all steps complete or are skipped

        The workflow provides optimal performance by executing independent steps
        in parallel while maintaining dependency order through completion events.

        Args:
            **kwargs: Execution parameters distributed to appropriate steps

        Yields:
            Events from step executions as they become available
        """
        if not self._steps:
            return

        queue = asyncio.Queue()
        completions = self._create_completion_events()
        tasks = self._create_step_tasks(queue, completions, **kwargs)

        try:
            async for event in self._process_event_queue(queue, len(tasks)):
                yield event
        except (asyncio.CancelledError, InterruptError):
            raise
        finally:
            await self._cleanup_tasks(tasks)
