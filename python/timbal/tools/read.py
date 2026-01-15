"""
Read tool for file access with path expansion support.

Returns a File object with content formatted for LLM consumption.
Supports ~ (home directory) and environment variables in paths.
"""
import hashlib
import os
from itertools import islice
from pathlib import Path

import structlog

from ..core.tool import Tool
from ..state import get_run_context
from ..types.file import File

logger = structlog.get_logger("timbal.tools.read")


def _resolve_path(path: str) -> Path:
    """Resolve a file path with context-aware security."""
    run_context = get_run_context()
    if run_context:
        return run_context.resolve_cwd(path)
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def _validate_line_range(start_line: int | None, end_line: int | None) -> None:
    """Validate line range parameters."""
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be >= 1 (1-indexed)")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be >= 1 (1-indexed)")
    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError(f"start_line ({start_line}) must be <= end_line ({end_line})")


def _update_file_state(path: Path, content: bytes | str) -> None:
    """Update file state tracking in run context."""
    run_context = get_run_context()
    if run_context and hasattr(run_context, "_fs_state"):
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content
        new_hash = hashlib.sha256(content_bytes).hexdigest()
        run_context._fs_state[str(path)] = new_hash


class Read(Tool):

    def __init__(self, **kwargs):
        
        async def _read(
            path: str,
            start_line: int | None = None,
            end_line: int | None = None
        ) -> File | str:
            """
            Read a file at the specified path.
            
            Args:
                path: Path to the file to read
                start_line: Optional starting line number (1-indexed, inclusive)
                end_line: Optional ending line number (1-indexed, inclusive)
            
            Returns:
                File object with content (optionally sliced to line range)
            """
            _validate_line_range(start_line, end_line)
            resolved_path = _resolve_path(path)

            if not resolved_path.exists():
                raise FileNotFoundError(f"File does not exist: {resolved_path}")
            
            if resolved_path.is_dir():
                return _read_directory(resolved_path)

            file = File.validate(resolved_path)
            _update_file_state(resolved_path, resolved_path.read_bytes())

            # Handle special file types that are not plain text
            if _is_special_file_type(file):
                return file

            # Read full file or specific line range
            if start_line is None and end_line is None:
                return file.read().decode("utf-8")

            return _read_line_range(resolved_path, start_line, end_line)
            
        super().__init__(
            name="read",
            description="Read a file at the specified path. Optionally specify start_line and end_line to read only a specific line range.",
            handler=_read,
            **kwargs
        )


def _read_directory(path: Path) -> str:
    """Read directory contents."""
    contents = "\n".join(item.name for item in path.iterdir())
    return contents if contents else "Empty directory"


def _is_special_file_type(file: File) -> bool:
    """Check if file is a special type that should be returned as File object."""
    special_extensions = [".xlsx", ".eml", ".docx"]
    special_content_types = ["image/", "audio/", "application/pdf"]
    
    if file.__source_extension__ in special_extensions:
        return True
    if any(file.__content_type__.startswith(ct) for ct in special_content_types):
        return True
    return False


def _read_line_range(path: Path, start_line: int | None, end_line: int | None) -> str:
    """Read a specific line range from a file."""
    with open(path, encoding="utf-8") as f:
        start_idx = (start_line - 1) if start_line is not None else 0
        
        if end_line is not None:
            if start_line is not None:
                num_lines = end_line - start_line + 1
            else:
                num_lines = end_line
        else:
            num_lines = None
        
        lines = list(islice(f, start_idx, start_idx + num_lines if num_lines else None))
    
    content = ''.join(lines)
    return content if content else ""
