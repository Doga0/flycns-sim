"""Small atomic artifact writer shared by bridge builds and offline replays."""

import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def stage_output(output: Path):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}-{uuid4().hex}"
    stage.mkdir()
    try:
        yield stage
        stage.rename(output)
    finally:
        if stage.exists() and stage.resolve().parent == output.parent:
            shutil.rmtree(stage)
