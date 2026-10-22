import pytest

from inspect_robots_evo.sandbox import SandboxError, check_source, limited_builtins


def test_math_and_json_are_allowed() -> None:
    check_source("import math\nimport json\n")


@pytest.mark.parametrize(
    "source",
    ["import os", "import subprocess", "from socket import socket"],
)
def test_other_imports_are_rejected(source: str) -> None:
    with pytest.raises(SandboxError):
        check_source(source)


def test_open_call_is_rejected() -> None:
    with pytest.raises(SandboxError):
        check_source("open('x')\n")


def test_runtime_import_of_os_is_rejected() -> None:
    namespace = {"__builtins__": limited_builtins()}
    with pytest.raises(SandboxError):
        exec("import os\n", namespace, namespace)
