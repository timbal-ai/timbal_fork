"""
Write tool for file creation and editing with path expansion support.

Creates new files or modifies existing ones with diff output.
Supports ~ (home directory) and environment variables in paths.
"""
import difflib
import hashlib
import os
from pathlib import Path

import structlog

from ..core.tool import Tool
from ..state import get_run_context

logger = structlog.get_logger("timbal.tools.write")


def _resolve_path(path: str) -> Path:
    """Resolve a file path with context-aware security."""
    run_context = get_run_context()
    if run_context:
        return run_context.resolve_cwd(path)
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def _validate_write_path(path: Path) -> None:
    """Validate that path is suitable for writing."""
    if path.exists() and path.is_dir():
        raise ValueError(f"Path is a directory, not a file: {path}")


def _read_existing_content(path: Path) -> str:
    """Read existing file content if file exists."""
    if not path.exists():
        return ""
    return path.read_bytes().decode("utf-8")


def _generate_diff(original_content: str, new_content: str, filename: str) -> str:
    """Generate a unified diff between original and new content."""
    diff_lines = list(difflib.unified_diff(
        original_content.splitlines(keepends=False),
        new_content.splitlines(keepends=False),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
        n=3
    ))
    return "\n".join(diff_lines)


def _update_file_state(path: Path, content: str) -> None:
    """Update file state tracking in run context."""
    run_context = get_run_context()
    if run_context and hasattr(run_context, "_fs_state"):
        new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        run_context._fs_state[str(path)] = new_hash


class Write(Tool):

    # TODO Add parameter to limit permissions to a specific path
    def __init__(self, **kwargs):

        async def _write(path: str, content: str) -> str:
            """
            Write content to a file, creating it if it doesn't exist or overwriting if it does.
            
            Creates parent directories automatically if they don't exist.
            Returns a unified diff showing the changes made.
            
            Args:
                path: Path to the file to write (supports ~ and environment variables)
                content: The complete content to write to the file
                
            Returns:
                Unified diff showing the changes made (empty for new files)
            """
            resolved_path = _resolve_path(path)
            _validate_write_path(resolved_path)

            original_content = _read_existing_content(resolved_path)
            
            # Create parent directories if they don't exist
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content, encoding="utf-8")

            _update_file_state(resolved_path, content)

            return _generate_diff(original_content, content, resolved_path.name)


        super().__init__(
            name="write",
            description=(
                "Write content to a file, creating it if it doesn't exist or overwriting if it does. "
                "Automatically creates parent directories. Returns a diff showing changes."
            ),
            handler=_write,
            **kwargs
        )
