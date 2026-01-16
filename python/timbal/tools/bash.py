"""
Bash tool for secure shell command execution with pattern validation.

Args:
    allowed_patterns: String or list of strings defining allowed command patterns.
                     Supports shell-style wildcards where '*' matches any sequence.
                     Use "*" to allow any command (use with caution).

Pattern Matching:
    - '*' in patterns is converted to regex that matches quoted strings or word characters
    - Command chains (&&, ||, |, ;) are validated by checking each part separately
    - Patterns are anchored (must match entire command)

Security Features:
    - Commands are validated against patterns before execution
    - Async subprocess execution with stdout/stderr capture
    - Return code and output tracking

Examples:
    Basic usage:
        Bash("echo *")              # Allow any echo command
        Bash(["ls *", "pwd"])       # Allow ls with args and pwd
        Bash("git status")          # Allow only exact git status

    Command chains:
        Bash("cd * && ls *")        # Allow cd followed by ls
        Bash("make && make test")   # Allow specific build sequence

    Wildcard patterns:
        Bash("python *.py")         # Allow python with .py files
        Bash("*")                   # Allow any command (dangerous)

Returns:
    Dict containing:
        - stdout: Command output as string
        - stderr: Error output as string
        - returncode: Process exit code

Warning:
    Pattern matching uses regex conversion and may not catch all edge cases.
    Complex shell syntax, escape sequences, or unusual command structures might
    bypass validation. Please submit issues or pull requests if you encounter
    commands that behave unexpectedly with the pattern matching system.
"""
import asyncio
import re
from pathlib import Path
from typing import Any

import structlog

from ..core.tool import Tool
from ..state import get_run_context

logger = structlog.get_logger("timbal.tools.bash")


class Bash(Tool):

    def __init__(self, allowed_patterns: str | list[str], **kwargs: Any):
        normalized_patterns = _normalize_patterns(allowed_patterns)
        compiled_patterns = _compile_patterns(normalized_patterns)

        async def _execute_command(command: str) -> dict[str, Any]:
            return await _execute_bash_command(command, compiled_patterns, normalized_patterns)

        super().__init__(
            name="bash",
            description=f"Execute a bash command. Allowed patterns: {normalized_patterns}",
            handler=_execute_command,
            background_mode="auto",
            **kwargs
        )

        self.allowed_patterns = normalized_patterns
        self.compiled_patterns = compiled_patterns

    def validate_command(self, command: str) -> dict[str, Any]:
        """Validate if a command matches any allowed pattern.
        
        Args:
            command: Command string to validate
            
        Returns:
            Dictionary containing:
            - is_allowed: Whether command is allowed
            - matched_pattern: Pattern that matched (if any)
            - errors: List of error messages
        """
        command = command.strip()
        errors = []
        
        # Check direct match
        for i, compiled_pattern in enumerate(self.compiled_patterns):
            if compiled_pattern.match(command):
                return {
                    "is_allowed": True,
                    "matched_pattern": self.allowed_patterns[i],
                    "errors": [],
                }
        
        # Check command chain
        chain_parts = re.split(r'\s*(?:\|\||\&\&|\||\;)\s*', command)
        for part in chain_parts:
            part_allowed = False
            for compiled_pattern in self.compiled_patterns:
                if compiled_pattern.match(part):
                    part_allowed = True
                    break
            if not part_allowed:
                errors.append(f"Command part '{part}' does not match any allowed pattern")
        
        return {
            "is_allowed": len(errors) == 0,
            "matched_pattern": None,
            "errors": errors,
        }

    def get_allowed_patterns(self) -> list[str]:
        """Get the list of allowed command patterns."""
        return self.allowed_patterns.copy()


def _normalize_patterns(allowed_patterns: str | list[str]) -> list[str]:
    """Normalize and validate pattern input."""
    if isinstance(allowed_patterns, str):
        allowed_patterns = [allowed_patterns]

    if not allowed_patterns:
        raise ValueError("At least one allowed pattern must be provided")

    return allowed_patterns


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Convert shell patterns to compiled regex patterns."""
    compiled_patterns = []
    
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError(f"Pattern must be a string, got {type(pattern)}")
        
        regex_pattern = pattern.strip()
        if not regex_pattern:
            raise ValueError("Pattern cannot be empty or whitespace only")

        # Special case: if pattern is just "*", accept everything
        if regex_pattern == "*":
            compiled_patterns.append(re.compile(r"^.*$"))
            continue

        compiled_regex = _build_regex_from_pattern(regex_pattern)
        compiled_patterns.append(compiled_regex)
    
    return compiled_patterns


def _build_regex_from_pattern(pattern: str) -> re.Pattern:
    """Build a regex pattern from a shell-style pattern."""
    parts = pattern.split()
    regex_parts = []
    
    for i, part in enumerate(parts):
        if "*" in part:
            # Wildcard: replace with argument pattern
            arg_pattern = r"""(?:(['"]).*?\1|[\w\/\\\-\.\,\*]+)(?:\s+(?:(['"]).*?\2|[\w\/\\\-\.\,\*]+))*"""
            if i > 0:
                regex_parts.append(r"(?:\s+" + arg_pattern + r")?")
            else:
                regex_parts.append(r"(?:" + arg_pattern + r")?")
        else:
            # Literal part: escape and add word boundary
            escaped = re.escape(part)
            if part[-1].isalnum():
                escaped = escaped + r"\b"
            
            if i > 0:
                regex_parts.append(r"\s+" + escaped)
            else:
                regex_parts.append(escaped)
    
    regex_pattern = "".join(regex_parts)
    regex_pattern = f"^{regex_pattern}$"
    return re.compile(regex_pattern)


def _validate_command_against_patterns(
    command: str, compiled_patterns: list[re.Pattern], allowed_patterns: list[str]
) -> None:
    """Validate command against compiled patterns."""
    command_allowed = False
    for compiled_pattern in compiled_patterns:
        if compiled_pattern.match(command):
            command_allowed = True
            break

    if not command_allowed:
        # Check command chains
        chain_parts = re.split(r'\s*(?:\|\||\&\&|\||\;)\s*', command)
        for part in chain_parts:
            part_allowed = False
            for compiled_pattern in compiled_patterns:
                if compiled_pattern.match(part):
                    part_allowed = True
                    break
            if not part_allowed:
                raise ValueError(
                    f"Command '{command}' does not match any allowed patterns: {allowed_patterns}"
                )


async def _execute_bash_command(
    command: str, compiled_patterns: list[re.Pattern], allowed_patterns: list[str]
) -> dict[str, Any]:
    """Execute a bash command with validation."""
    command = command.strip()
    _validate_command_against_patterns(command, compiled_patterns, allowed_patterns)

    # Resolve working directory
    run_context = get_run_context()
    if run_context:
        cwd = run_context.resolve_cwd()
    else:
        cwd = Path.cwd()

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )

    stdout, stderr = await process.communicate()
    stdout = stdout.decode("utf-8") if stdout else ""
    stderr = stderr.decode("utf-8") if stderr else ""

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": process.returncode,
    }

