"""Built-in MCP server exposing deterministic engineering utilities."""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP(
    "project-advisor-utilities",
    instructions=(
        "Use these tools for deterministic cost and license checks while evaluating "
        "AI application frameworks. Do not estimate these values mentally."
    ),
)


class CostEstimate(BaseModel):
    monthly_cost_usd: float
    request_cost_usd: float
    monthly_input_tokens: int
    monthly_output_tokens: int
    assumptions: list[str]


@mcp.tool(structured_output=True)
def estimate_llm_cost(
    monthly_requests: int = Field(ge=1, description="Expected requests per month"),
    average_input_tokens: int = Field(ge=0, description="Average input tokens per request"),
    average_output_tokens: int = Field(ge=0, description="Average output tokens per request"),
    input_price_per_million: float = Field(ge=0, description="Input price in USD per one million tokens"),
    output_price_per_million: float = Field(ge=0, description="Output price in USD per one million tokens"),
) -> CostEstimate:
    """Calculate reproducible per-request and monthly LLM token cost."""
    monthly_input_tokens = monthly_requests * average_input_tokens
    monthly_output_tokens = monthly_requests * average_output_tokens
    monthly_cost = (
        monthly_input_tokens * input_price_per_million
        + monthly_output_tokens * output_price_per_million
    ) / 1_000_000

    return CostEstimate(
        monthly_cost_usd=round(monthly_cost, 4),
        request_cost_usd=round(monthly_cost / monthly_requests, 6),
        monthly_input_tokens=monthly_input_tokens,
        monthly_output_tokens=monthly_output_tokens,
        assumptions=[
            "Prices are supplied by the caller and should be checked against current vendor pricing.",
            "Caching, retries, embeddings, reranking, and infrastructure are excluded.",
        ],
    )


@mcp.tool(structured_output=True)
def check_license_policy(
    license_id: str = Field(description="SPDX-like license identifier, for example MIT or AGPL-3.0"),
    commercial_use: bool = Field(description="Whether the project will be used commercially"),
    closed_source_distribution: bool = Field(description="Whether modified software will be distributed as closed source"),
) -> dict:
    """Apply a conservative license-policy check for project shortlisting."""
    normalized = license_id.strip().upper()
    permissive = {"MIT", "APACHE-2.0", "BSD-2-CLAUSE", "BSD-3-CLAUSE", "ISC"}
    strong_copyleft = {"AGPL-3.0", "GPL-3.0", "GPL-2.0"}

    if normalized in permissive:
        risk = "low"
        decision = "generally_compatible"
        reason = "Permissive license; retain notices and verify third-party dependencies."
    elif normalized in strong_copyleft and commercial_use and closed_source_distribution:
        risk = "high"
        decision = "legal_review_required"
        reason = "Copyleft obligations may conflict with closed-source distribution."
    else:
        risk = "medium"
        decision = "legal_review_required"
        reason = "License policy is not covered by the deterministic allowlist."

    return {
        "license_id": normalized,
        "risk": risk,
        "decision": decision,
        "reason": reason,
        "disclaimer": "This engineering check is not legal advice.",
    }


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
