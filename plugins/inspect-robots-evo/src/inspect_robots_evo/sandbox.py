"""提交代码只能 import math 和 json, 也不能调用 open."""

from __future__ import annotations

import ast


class SandboxError(Exception):
    """代码在执行前或 import 时被拒绝."""


_ALLOWED = {"math", "json"}
_BANNED_CALLS = {"open", "__import__", "exec", "eval", "compile"}


def check_source(source: str) -> None:
    """拒绝 math 和 json 以外的 import, 以及 open 等调用."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _allow(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise SandboxError("relative import is not allowed")
            _allow(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _BANNED_CALLS
        ):
            raise SandboxError(f"call {node.func.id} is not allowed")


def limited_builtins() -> dict[str, object]:
    """返回只允许 math 和 json 的 builtins."""

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise SandboxError("relative import is not allowed")
        _allow(name)
        return __import__(name, globals, locals, fromlist, level)

    return {
        "False": False,
        "None": None,
        "True": True,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": _import,
    }


def _allow(name: str) -> None:
    if name.split(".", 1)[0] not in _ALLOWED:
        raise SandboxError(f"import {name} is not allowed")
