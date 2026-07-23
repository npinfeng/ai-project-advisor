"""MCP client configuration and LangChain tool loading."""

import asyncio
import json
import logging
import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from project_advisor.configuration import Configuration

logger = logging.getLogger(__name__)

_tool_cache: dict[str, list[BaseTool]] = {}
_client_cache: dict[str, MultiServerMCPClient] = {}
_connection_state: dict[str, dict[str, Any]] = {}


def build_mcp_connections(config: Configuration) -> dict[str, dict[str, Any]]:
    """Build stdio/HTTP MCP connections from typed configuration."""
    connections: dict[str, dict[str, Any]] = {}

    if config.enable_local_mcp:
        connections["advisor_utilities"] = {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "project_advisor.mcp_server"],
        }

    if config.mcp_servers_json.strip():
        external = json.loads(config.mcp_servers_json)
        if not isinstance(external, dict):
            raise ValueError("MCP_SERVERS_JSON 必须是以服务名为键的 JSON 对象。")
        for server_name, connection in external.items():
            if not isinstance(connection, dict) or "transport" not in connection:
                raise ValueError(f"MCP Server '{server_name}' 缺少有效连接配置。")
            connections[server_name] = connection

    return connections


async def get_mcp_tools(
    config: Configuration,
    *,
    force_refresh: bool = False,
) -> list[BaseTool]:
    """Connect to configured MCP servers and return executable LangChain tools."""
    connections = build_mcp_connections(config)
    if not connections:
        return []

    cache_key = json.dumps(connections, ensure_ascii=False, sort_keys=True, default=str)
    if not force_refresh and cache_key in _tool_cache:
        return _tool_cache[cache_key]

    _connection_state[cache_key] = {
        "status": "connecting",
        "server_count": len(connections),
        "tool_count": 0,
        "error_type": None,
    }
    try:
        client = MultiServerMCPClient(
            connections,
            tool_name_prefix=True,
            handle_tool_errors=True,
        )
        tools = await asyncio.wait_for(
            client.get_tools(),
            timeout=config.mcp_connect_timeout_seconds,
        )
    except Exception as error:
        _connection_state[cache_key] = {
            "status": "degraded",
            "server_count": len(connections),
            "tool_count": 0,
            "error_type": type(error).__name__,
        }
        if config.mcp_required:
            raise RuntimeError("无法连接必需的 MCP Server。") from error
        logger.warning("MCP tools unavailable: %s", error)
        return []

    _client_cache[cache_key] = client
    _tool_cache[cache_key] = tools
    _connection_state[cache_key] = {
        "status": "connected",
        "server_count": len(connections),
        "tool_count": len(tools),
        "error_type": None,
    }
    return tools


def get_mcp_diagnostics(config: Configuration) -> dict[str, Any]:
    """Return non-secret MCP connection diagnostics without opening a connection."""
    try:
        connections = build_mcp_connections(config)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid_configuration",
            "server_count": 0,
            "tool_count": 0,
            "error_type": type(error).__name__,
        }
    if not connections:
        return {
            "status": "disabled",
            "server_count": 0,
            "tool_count": 0,
            "error_type": None,
        }

    cache_key = json.dumps(connections, ensure_ascii=False, sort_keys=True, default=str)
    return _connection_state.get(
        cache_key,
        {
            "status": "configured",
            "server_count": len(connections),
            "tool_count": len(_tool_cache.get(cache_key, [])),
            "error_type": None,
        },
    ).copy()


def clear_mcp_tool_cache() -> None:
    """Clear MCP client caches for tests and configuration reloads."""
    _tool_cache.clear()
    _client_cache.clear()
    _connection_state.clear()
