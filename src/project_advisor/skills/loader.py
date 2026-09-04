"""Load repository-owned SKILL.md guidance into the appropriate researcher."""

from functools import lru_cache
from pathlib import Path


SKILLS_BY_ROLE: dict[str, tuple[str, ...]] = {
    "repository": (
        "repository-due-diligence",
        "dependency-license-audit",
    ),
    "documentation": (
        "official-doc-verification",
        "rag-architecture-review",
    ),
}


def _skills_root() -> Path:
    return Path(__file__).resolve().parent


def _body(markdown: str) -> str:
    """Remove YAML frontmatter; the model only needs the operational guidance."""
    if not markdown.startswith("---"):
        return markdown.strip()
    parts = markdown.split("---", 2)
    return parts[2].strip() if len(parts) == 3 else markdown.strip()


@lru_cache(maxsize=16)
def load_skill(name: str) -> str:
    """Load a validated, local skill by its directory name."""
    if name not in {skill for values in SKILLS_BY_ROLE.values() for skill in values}:
        raise ValueError(f"未知 Agent Skill：{name}")
    path = _skills_root() / name / "SKILL.md"
    return _body(path.read_text(encoding="utf-8"))


def load_skills_for_role(role: str) -> str:
    """Render only the skills assigned to a researcher role."""
    names = SKILLS_BY_ROLE.get(role, ())
    if not names:
        return ""
    sections = [f"<skill name=\"{name}\">\n{load_skill(name)}\n</skill>" for name in names]
    return "\n\n<role_skills>\n" + "\n\n".join(sections) + "\n</role_skills>"
