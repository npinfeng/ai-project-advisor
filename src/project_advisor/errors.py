"""Domain exception hierarchy for stable API and workflow error handling."""

from __future__ import annotations


class ProjectAdvisorError(Exception):
    """Base class for errors the application can classify intentionally."""


class ModelError(ProjectAdvisorError):
    """Base class for model configuration or invocation failures."""


class ModelConfigurationError(ModelError, ValueError):
    """The selected provider cannot be created from current configuration."""


class StructuredOutputError(ModelError, RuntimeError):
    """A model exhausted its bounded attempts without valid structured output."""


class ToolExecutionError(ProjectAdvisorError):
    """Base class for failures raised while invoking a tool."""


class ToolReportedError(ToolExecutionError, RuntimeError):
    """A tool returned an error observation instead of raising an exception."""

    def __init__(self, message: str, *, retryable: bool, timed_out: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.timed_out = timed_out


class AgentRunTimeoutError(ProjectAdvisorError, RuntimeError):
    """An Agent Run exhausted its end-to-end execution deadline."""


class PersistenceError(ProjectAdvisorError, RuntimeError):
    """Persistent task or checkpoint state could not be read or written."""
