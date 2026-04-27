import json
from pathlib import Path

from src.eval_app.models import RunRecord


def load_runs(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RunRecord.model_validate(item) for item in data]


def save_runs(path: Path, runs: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [run.model_dump(mode="json") for run in runs]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


def upsert_run(path: Path, new_run: RunRecord) -> tuple[list[RunRecord], bool]:
    runs = load_runs(path)
    replaced = False
    for idx, run in enumerate(runs):
        if run.run_name == new_run.run_name:
            runs[idx] = new_run
            replaced = True
            break
    if not replaced:
        runs.append(new_run)
    save_runs(path, runs)
    return runs, replaced


def get_latest_run(path: Path) -> RunRecord | None:
    runs = load_runs(path)
    if not runs:
        return None
    return sorted(runs, key=lambda run: run.updated_at)[-1]
