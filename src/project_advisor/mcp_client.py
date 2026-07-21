"""MCP client configuration and LangChain tool loading."""

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

    try:
        client = MultiServerMCPClient(
            connections,
            tool_name_prefix=True,
            handle_tool_errors=True,
        )
        tools = await client.get_tools()
    except Exception as error:
        if config.mcp_required:
            raise RuntimeError("无法连接必需的 MCP Server。") from error
        logger.warning("MCP tools unavailable: %s", error)
        return []

    _client_cache[cache_key] = client
    _tool_cache[cache_key] = tools
    return tools


def clear_mcp_tool_cache() -> None:
    """Clear MCP client caches for tests and configuration reloads."""
    _tool_cache.clear()
    _client_cache.clear()
