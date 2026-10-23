import json
from pathlib import Path

from inspect_robots_pai.evolution.contract import passed, promote
from inspect_robots_pai.memory.scaffold import scaffold
from inspect_robots_pai.memory.store import lookup_task


def _write_diag(
    trial_dir: Path,
    *,
    process_ok: bool = True,
    success: bool | None = None,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"process_ok": process_ok}
    if success is not None:
        record["success"] = success
    (trial_dir / "diagnostics.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def test_unlabeled_does_not_pass(tmp_path: Path) -> None:
    run = tmp_path / "run"
    trial = run / "trials" / "0"
    _write_diag(trial, process_ok=True)
    (trial / "auto_check.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (run / "note.txt").write_text("CONVERGED", encoding="utf-8")

    assert passed(run, 1) is False
    library = tmp_path / "library"
    assert not library.exists()


def test_failing_promote_does_not_copy(tmp_path: Path) -> None:
    run = tmp_path / "run"
    candidate = run / "candidate"
    scaffold(candidate, "bottle")
    _write_diag(run / "trials" / "0", process_ok=True, success=False)
    library = tmp_path / "library"
    memory = tmp_path / "memory"
    promote(
        run,
        candidate,
        library,
        "bottle",
        memory,
        "pick bottle",
        "missed grasp",
        "fix",
    )
    assert not (library / "bottle").exists()


def test_success_promotes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    candidate = run / "candidate"
    scaffold(candidate, "bottle")
    description = "put the bottle on the pad"
    _write_diag(run / "trials" / "0", process_ok=True, success=True)
    library = tmp_path / "library"
    memory = tmp_path / "memory"
    promote(
        run,
        candidate,
        library,
        "bottle",
        memory,
        description,
        "failed to grasp",
        "fix",
    )

    lib_init = library / "bottle" / "__init__.py"
    assert lib_init.is_file()
    skills = json.loads(
        (memory / "skills" / "approach.json").read_text(encoding="utf-8"),
    )
    assert skills[0]["repair"] == "fix"
    assert skills[0]["code_ref"] == "code_library/nero/bottle"
    tasks = json.loads((memory / "tasks.json").read_text(encoding="utf-8"))
    program_path = lookup_task(memory, description)
    assert program_path == library / "bottle"
    assert tasks[description] == str(library / "bottle")

    original = lib_init.read_text(encoding="utf-8")
    (candidate / "__init__.py").write_text("# mutated\n", encoding="utf-8")
    assert lib_init.read_text(encoding="utf-8") == original
