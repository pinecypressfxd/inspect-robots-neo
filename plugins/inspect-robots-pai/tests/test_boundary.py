from pathlib import Path

ROOT = Path("plugins/inspect-robots-pai/src/inspect_robots_pai")
FORBIDDEN = (
    "inspect_robots_evo",
    "inspect_robots_harness",
    "inspect_robots_vla",
    "inspect_robots_nero",
    "inspect_robots_agent",
)


def test_package_does_not_import_other_plugins():
    offenders = []
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN:
            if name in text:
                offenders.append(str(path) + ":" + name)
        if "domains.nero" in text and path.name != "policy.py":
            offenders.append(str(path) + ":domains.nero")
    assert offenders == []
