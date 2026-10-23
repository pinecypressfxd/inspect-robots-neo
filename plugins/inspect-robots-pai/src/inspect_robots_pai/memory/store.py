import json
from pathlib import Path


def write_task(memory_root: Path, description: str, program: Path) -> None:
    memory_root = Path(memory_root)
    memory_root.mkdir(parents=True, exist_ok=True)
    path = memory_root / "tasks.json"
    tasks: dict[str, str] = {}
    if path.is_file():
        tasks = json.loads(path.read_text(encoding="utf-8"))
    tasks[description] = str(program)
    path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")


def write_skill(
    memory_root: Path,
    failure: str,
    repair: str,
    task: str,
) -> None:
    memory_root = Path(memory_root)
    skills_dir = memory_root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    path = skills_dir / "approach.json"
    entries: list[dict] = []
    if path.is_file():
        entries = json.loads(path.read_text(encoding="utf-8"))
    entries.append(
        {
            "failure": failure,
            "repair": repair,
            "code_ref": f"code_library/nero/{task}",
        },
    )
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def lookup_task(memory_root: Path, description: str) -> Path | None:
    path = Path(memory_root) / "tasks.json"
    if not path.is_file():
        return None
    tasks = json.loads(path.read_text(encoding="utf-8"))
    stored = tasks.get(description)
    if stored is None:
        return None
    return Path(stored)
