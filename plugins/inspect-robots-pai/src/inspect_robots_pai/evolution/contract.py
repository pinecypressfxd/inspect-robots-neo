import json
import shutil
from pathlib import Path

from inspect_robots_pai.memory.store import write_skill, write_task


def _trial_count(run_dir: Path) -> int:
    contract = Path(run_dir) / "contract.json"
    if contract.is_file():
        data = json.loads(contract.read_text(encoding="utf-8"))
        if "trial_count" in data:
            return int(data["trial_count"])
    return 1


def passed(run_dir: Path, trial_count: int) -> bool:
    run_dir = Path(run_dir)
    for index in range(trial_count):
        diag_path = run_dir / "trials" / str(index) / "diagnostics.jsonl"
        if not diag_path.is_file():
            return False
        line = diag_path.read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(line)
        if not record.get("process_ok"):
            return False
        if "success" not in record or not record["success"]:
            return False
    return True


def promote(
    run_dir: Path,
    candidate: Path,
    library_root: Path,
    task: str,
    memory_root: Path,
    description: str,
    failure: str,
    repair: str,
) -> None:
    if not passed(run_dir, _trial_count(run_dir)):
        return
    program = Path(library_root) / task
    shutil.copytree(candidate, program, dirs_exist_ok=True)
    write_task(memory_root, description, program)
    write_skill(memory_root, failure, repair, task)
