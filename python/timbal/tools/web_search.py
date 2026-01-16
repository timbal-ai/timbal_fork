"""
WebSearch is a specification-only tool. 

Specification-only tools are tools that exist purely to define parameter
schemas and return types for LLM interaction, without containing any
executable logic. They serve as "interface contracts" that tell the LLM what
parameters are expected and what the tool conceptually returns, but the
actual execution happens elsewhere (or not at all).
"""
from functools import cached_property
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import computed_field

from ..core.tool import Tool

logger = structlog.get_logger("timbal.tools.web_search")


class WebSearch(Tool):

    def __init__(
        self, 
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict[str, Any] | None = None,
        # ? anthropic's max_uses
        **kwargs: Any,
    ) -> None:

        def _unreachable_handler():
            raise NotImplementedError("This is a specification-only tool")

        _validate_web_search_config(allowed_domains, blocked_domains, user_location)

        super().__init__(
            name="web_search",
            handler=_unreachable_handler,
            **kwargs
        )

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        # User location example: {
        #     "type": "approximate",
        #     "country": "GB",
        #     "city": "London",
        #     "region": "London",
        #     "timezone": "Europe/London"
        # }
        self.user_location = user_location

    def get_configuration(self) -> dict[str, Any]:
        """Get the current web search configuration.
        
        Returns:
            Dictionary containing configuration settings
        """
        return {
            "allowed_domains": self.allowed_domains,
            "blocked_domains": self.blocked_domains,
            "user_location": self.user_location,
        }

    def validate_configuration(self) -> dict[str, Any]:
        """Validate the web search configuration.
        
        Returns:
            Dictionary containing validation results
        """
        errors = []
        warnings = []
        
        if self.allowed_domains and self.blocked_domains:
            # Check for overlap
            overlap = set(self.allowed_domains) & set(self.blocked_domains)
            if overlap:
                errors.append(f"Domains cannot be both allowed and blocked: {overlap}")
        
        if self.allowed_domains:
            for domain in self.allowed_domains:
                if not _is_valid_domain(domain):
                    errors.append(f"Invalid domain format in allowed_domains: {domain}")
        
        if self.blocked_domains:
            for domain in self.blocked_domains:
                if not _is_valid_domain(domain):
                    errors.append(f"Invalid domain format in blocked_domains: {domain}")
        
        if self.user_location:
            required_fields = ["type", "country"]
            for field in required_fields:
                if field not in self.user_location:
                    warnings.append(f"User location missing recommended field: {field}")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


def _validate_web_search_config(
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
    user_location: dict[str, Any] | None,
) -> None:
    """Validate web search configuration parameters."""
    if allowed_domains is not None and not isinstance(allowed_domains, list):
        raise TypeError("allowed_domains must be a list or None")
    
    if blocked_domains is not None and not isinstance(blocked_domains, list):
        raise TypeError("blocked_domains must be a list or None")
    
    if user_location is not None and not isinstance(user_location, dict):
        raise TypeError("user_location must be a dict or None")


def _is_valid_domain(domain: str) -> bool:
    """Basic domain validation."""
    if not domain or not isinstance(domain, str):
        return False
    # Basic check: domain should contain at least one dot and no spaces
    return "." in domain and " " not in domain and len(domain) > 0

    
    @override
    @computed_field
    @cached_property
    def openai_responses_schema(self) -> dict[str, Any]:
        """See base class."""
        schema = {"type": "web_search",}

        if self.allowed_domains:
            schema["filters"] = {"allowed_domains": self.allowed_domains}
        if self.blocked_domains:
            logger.warning("Blocked domains are not supported by OpenAI.")
        if self.user_location:
            schema["user_location"] = self.user_location

        return schema
    

    @override
    @computed_field(repr=False)
    @cached_property
    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """See base class."""
        raise ValueError("WebSearch is not compatible with OpenAI's chat completions API.")


    @override
    @computed_field
    @cached_property
    def anthropic_schema(self) -> dict[str, Any]:
        """See base class."""
        anthropic_schema = {
            "type": "web_search_20250305", # TODO Review
            "name": "web_search",
        }

        if self.allowed_domains:
            anthropic_schema["allowed_domains"] = self.allowed_domains
        if self.blocked_domains:
            anthropic_schema["blocked_domains"] = self.blocked_domains
        if self.user_location:
            anthropic_schema["user_location"] = self.user_location

        return anthropic_schema
    