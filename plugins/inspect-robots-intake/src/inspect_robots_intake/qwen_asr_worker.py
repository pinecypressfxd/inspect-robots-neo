"""Keep one local Qwen3-ASR model loaded and transcribe wav paths from stdin.

Run this with the dedicated qwen-asr environment, not the mission-console
environment. The console writes one wav path per line and reads one JSON line
back. The first stdout line is ``ready`` after the weights are loaded.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    """Load ``sys.argv[1]`` and serve transcription requests until stdin closes."""
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        sys.argv[1],
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=128,
    )
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            results = model.transcribe(audio=path, language="Chinese")
            text = str(results[0].text).strip()
            payload = {"text": text}
        except Exception as exc:  # one bad clip must not kill the worker
            payload = {"error": str(exc)}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
