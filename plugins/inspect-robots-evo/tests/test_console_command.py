"""The mission console argv for an evo run."""

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path


def _console():
    path = Path("plugins/inspect-robots-nero/scripts/mission_console.py")
    return SourceFileLoader("mission_console_evo", str(path)).load_module()


def test_evo_command_uses_unlimited_steps_and_the_workspace() -> None:
    console = _console()
    args = argparse.Namespace(
        policy="evo",
        model="mog-10-a",
        base_url="https://kspmas.ksyun.com/v1",
        api_key_env="JINSHANYUN_CHATGPT_KEY",
        max_speed_frac=0.05,
        evo_mode="playground",
        evo_workspace="/tmp/evo-task",
    )
    _prefix, suffix = console.build_command(args)
    assert suffix[suffix.index("--policy") + 1] == "evo"
    assert suffix[suffix.index("--max-steps") + 1] == "0"
    assert "mode=playground" in suffix
    assert "workspace=/tmp/evo-task" in suffix
    assert not any(item.startswith("max_speed_frac=") for item in suffix)
