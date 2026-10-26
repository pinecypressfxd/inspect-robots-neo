"""一个任务一个目录。write_file 是唯一写入口。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class WorkspaceError(ValueError):
    """调用方写到了这个模式不允许的路径。"""


_PRIMITIVES = """\
get_instruction() -> str
get_observation() -> dict with left_rgbd, right_rgbd, chest_rgbd, eef_state
step(action) -> dict with held and steps. limited lists any z the runtime raised.
action is an absolute end-effector target. Unlisted dimensions hold.
No reset().
Labels: left_x, left_y, left_z, left_r1..left_r6, left_gripper, and the right arm.
Meters. Gripper 0 closed, 0.09 open.
The gripper extends 0.03 m below the TCP. A horizontal move through a rim or a
standing object snags. Boxes, cups, blocks, and objects just placed are obstacles.
When one step changes xy and z, the runtime raises z first, translates xy at
constant z, then lowers z. Cruise at the higher of the current z, the target z,
and this trial's start z. Lower only after xy is above the landing point.
The start z is about 0.05 m above the box rim. Do not command z more than
0.05 m below that start z. The runtime clamps a lower target and names it in limited.
A straight-down orientation is reachable only over part of the workspace.
held true and steps 0 means the arm did not reach. Change the target.
scripts/env.py step(code) runs submission/solution.py. solve() must return str.
"""

_ENV_PY = '''\
"""Host execution client. solution.py must not import this module.

step(code) executes submission/solution.py. The coding agent calls step_code.
Inside the program, step(action) is one absolute end-effector target.
"""


def step(code: str) -> dict:
    """The evo host executes this. Calling the file directly does nothing."""
    raise RuntimeError("step(code) is provided by the evo host")
'''


class Workspace:
    """一个任务目录. write_file 按模式限制可写路径."""

    def __init__(self, root: Path, mode: str) -> None:
        if mode not in ("playground", "test"):
            raise WorkspaceError(f"unknown mode {mode}")
        self.root = root
        self._root_resolved = root.resolve()
        self.mode = mode

    @classmethod
    def create(cls, parent: Path, instruction: str, *, mode: str, stamp: str) -> Workspace:
        """建立任务说明, 原语文档和空的提交文件."""
        parent_resolved = parent.resolve()
        root = parent / stamp
        root_resolved = root.resolve()
        if parent_resolved not in root_resolved.parents:
            raise WorkspaceError(f"stamp escapes parent: {stamp}")
        root.mkdir(parents=True)
        workspace = cls(root, mode)
        workspace._ensure_layout(instruction)
        return workspace

    def prepare(self, instruction: str) -> None:
        """补齐控制台已建好的空目录. 已有的 instruction.md 保持不变."""
        self._ensure_layout(instruction)

    def _ensure_layout(self, instruction: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        instruction_path = self.root / "instruction.md"
        if not instruction_path.exists():
            instruction_path.write_text(instruction, encoding="utf-8")
        primitives = self.root / "primitives" / "primitives.md"
        if not primitives.exists():
            primitives.parent.mkdir(parents=True, exist_ok=True)
            primitives.write_text(_PRIMITIVES, encoding="utf-8")
        env_py = self.root / "scripts" / "env.py"
        if not env_py.exists():
            env_py.parent.mkdir(parents=True, exist_ok=True)
            env_py.write_text(_ENV_PY, encoding="utf-8")
        submission = self.root / "submission"
        submission.mkdir(exist_ok=True)
        solution = submission / "solution.py"
        if not solution.exists():
            solution.write_text("", encoding="utf-8")
        for name in ("skills", "lessons", "scratch", "observations"):
            (self.root / name).mkdir(exist_ok=True)

    def write_file(self, relative: str, text: str) -> None:
        """写入白名单内的相对路径."""
        path = self._writable(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read_file(self, relative: str) -> str:
        """读取工作区里的一个文件."""
        path = self._inside(relative)
        if not path.is_file():
            raise WorkspaceError(f"no file {relative}")
        return path.read_text(encoding="utf-8")

    def freeze(self) -> None:
        """只复制 skills 和 lessons."""
        dest = self.root / "freeze"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir()
        shutil.copytree(self.root / "skills", dest / "skills")
        shutil.copytree(self.root / "lessons", dest / "lessons")

    def seed_from(self, freeze_dir: Path) -> None:
        """用一份冻结结果替换当前 skills 和 lessons."""
        skills_src = freeze_dir / "skills"
        lessons_src = freeze_dir / "lessons"
        if not skills_src.is_dir() or not lessons_src.is_dir():
            raise WorkspaceError("freeze directory missing skills or lessons")
        for name in ("skills", "lessons"):
            target = self.root / name
            shutil.rmtree(target)
            shutil.copytree(freeze_dir / name, target)

    def record_outcome(self, success: bool) -> None:
        """记录操作员给出的成功或失败."""
        payload = json.dumps({"success": success})
        (self.root / "outcome.json").write_text(payload, encoding="utf-8")

    def _writable(self, relative: str) -> Path:
        path = self._inside(relative)
        rel = path.relative_to(self._root_resolved).as_posix()
        locked = rel in ("instruction.md", "task.md", "primitives.md")
        locked = locked or rel.startswith(
            ("primitives/", "scripts/", "observations/", "submits/", "freeze/")
        )
        if locked:
            raise WorkspaceError(f"cannot write {rel}")
        allowed = rel == "submission/solution.py" or rel.startswith("scratch/")
        if self.mode == "playground":
            allowed = allowed or rel.startswith(("skills/", "lessons/"))
        if not allowed:
            raise WorkspaceError(f"cannot write {rel}")
        return path

    def _inside(self, relative: str) -> Path:
        root = self._root_resolved
        path = (self.root / relative).resolve()
        if path != root and root not in path.parents:
            raise WorkspaceError(f"path escapes workspace: {relative}")
        return path
