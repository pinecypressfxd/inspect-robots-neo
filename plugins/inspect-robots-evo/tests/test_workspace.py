from pathlib import Path

import pytest

from inspect_robots_evo.workspace import Workspace, WorkspaceError


def test_create_lays_out_the_task_directory(tmp_path: Path) -> None:
    workspace = Workspace.create(
        tmp_path, "pick up the cup", mode="playground", stamp="20261015-170000_aa"
    )
    assert (workspace.root / "task.md").read_text(encoding="utf-8") == "pick up the cup"
    assert "move_ee" in (workspace.root / "primitives.md").read_text(encoding="utf-8")
    assert (workspace.root / "submission" / "solution.py").read_text(encoding="utf-8") == ""
    assert (workspace.root / "skills").is_dir()
    assert (workspace.root / "lessons").is_dir()
    assert (workspace.root / "scratch").is_dir()


def test_playground_write_accepts_skills_and_rejects_task(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "task", mode="playground", stamp="s")
    workspace.write_file("skills/pick.py", "def pick():\n    return 1\n")
    workspace.write_file("lessons/grip.md", "close before lift\n")
    with pytest.raises(WorkspaceError, match="task.md"):
        workspace.write_file("task.md", "changed")


def test_test_mode_rejects_skill_writes(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "task", mode="test", stamp="s")
    with pytest.raises(WorkspaceError, match="skills"):
        workspace.write_file("skills/pick.py", "def pick():\n    return 1\n")


def test_freeze_copies_only_skills_and_lessons(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "task", mode="playground", stamp="s")
    workspace.write_file("skills/pick.py", "def pick():\n    return 1\n")
    workspace.write_file("lessons/grip.md", "close before lift\n")
    workspace.write_file("scratch/note.txt", "draft\n")
    workspace.freeze()
    freeze = workspace.root / "freeze"
    assert (freeze / "skills" / "pick.py").is_file()
    assert (freeze / "lessons" / "grip.md").is_file()
    assert not (freeze / "scratch").exists()


def test_seed_copies_freeze_then_locks_those_dirs(tmp_path: Path) -> None:
    source = Workspace.create(tmp_path / "src", "task", mode="playground", stamp="s")
    source.write_file("skills/pick.py", "def pick():\n    return 1\n")
    source.freeze()
    target = Workspace.create(tmp_path / "dst", "task", mode="test", stamp="t")
    target.seed_from(source.root / "freeze")
    assert (target.root / "skills" / "pick.py").is_file()
    with pytest.raises(WorkspaceError):
        target.write_file("skills/pick.py", "def pick():\n    return 2\n")
