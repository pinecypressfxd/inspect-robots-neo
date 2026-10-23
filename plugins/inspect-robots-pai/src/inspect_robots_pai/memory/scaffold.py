from pathlib import Path


def scaffold(dest: Path, task: str) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    source = f'''TASK_NAME = {task!r}

def run(port, trial_dir):
    import json
    from pathlib import Path

    phases = [{{"name": "scaffold"}}]
    (Path(trial_dir) / "phases.json").write_text(
        json.dumps(phases),
        encoding="utf-8",
    )
'''
    (dest / "__init__.py").write_text(source, encoding="utf-8")
