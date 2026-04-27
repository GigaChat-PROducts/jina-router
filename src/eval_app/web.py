import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run src/eval_app/web.py` from arbitrary cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_app.models import RunConfig, RunRecord
from src.eval_app.runner import run_named_evaluation
from src.eval_app.store import get_latest_run, load_runs
from src.evaluation.evaluator import build_default_config
from src.evaluation.quotas import QuotaTask

RUNS_PATH = Path("src/evaluation/data/runs.json")
DATASET_PATH = Path("src/dataset/data/labeled_dataset.json")


def _default_run_config() -> RunConfig:
    latest = get_latest_run(RUNS_PATH)
    if latest is not None:
        return latest.config
    return RunConfig.from_eval_config(build_default_config())


def _tasks_from_json(raw: str) -> list[QuotaTask]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("quota_tasks must be a JSON array")
    return [QuotaTask.model_validate(item) for item in data]


def _records_to_plot(records: list[RunRecord], x_field: str, y_field: str, title: str):
    if not records:
        st.info("No runs yet")
        return None

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[getattr(record, x_field) for record in records],
                y=[getattr(record, y_field) for record in records],
                mode="markers+text",
                text=[record.run_name for record in records],
                textposition="top center",
                customdata=[[record.run_name] for record in records],
                marker={"size": 11},
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title=x_field, yaxis_title=y_field, height=420)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    return event


def _extract_selected_run(event, records: list[RunRecord]) -> str | None:
    if not event:
        return None
    selection = None
    if isinstance(event, dict):
        selection = event.get("selection")
    elif hasattr(event, "selection"):
        selection = getattr(event, "selection")
    if not selection:
        return None
    points = []
    if isinstance(selection, dict):
        points = selection.get("points", [])
    elif hasattr(selection, "points"):
        points = getattr(selection, "points")
    if not points:
        return None
    point = points[0]
    custom_data = point.get("customdata")
    if isinstance(custom_data, list) and custom_data:
        return str(custom_data[0])
    point_index = point.get("point_index")
    if isinstance(point_index, int) and 0 <= point_index < len(records):
        return records[point_index].run_name
    return None


@st.dialog("Run details")
def _show_run_dialog(record: RunRecord) -> None:
    st.subheader(record.run_name)
    st.json(record.model_dump(mode="json"))


def main() -> None:
    st.set_page_config(page_title="Router eval", layout="wide")
    st.title("Router evaluation")

    default = _default_run_config()
    with st.form("run_form"):
        run_name = st.text_input("Run name", value="")
        reranker_endpoint = st.text_input(
            "Reranker endpoint", value=default.reranker_endpoint
        )
        global_token_limit = st.number_input(
            "Global token limit",
            min_value=1,
            step=100,
            value=int(default.global_token_limit),
        )
        enable_quota_reranking = st.checkbox(
            "Enable quota reranking", value=default.enable_quota_reranking
        )
        task_temp = st.number_input(
            "Task temperature",
            value=float(default.task_reranking_temperature),
            min_value=0.001,
            step=0.1,
        )
        product_temp = st.number_input(
            "Product temperature",
            value=float(default.product_reranking_temperature),
            min_value=0.001,
            step=0.1,
        )
        base_multiplier = st.number_input(
            "Base product multiplier",
            value=float(default.base_product_multiplier),
            min_value=0.001,
            step=0.1,
        )
        current_multiplier = st.number_input(
            "Current product multiplier",
            value=float(default.current_product_multiplier),
            min_value=0.001,
            step=0.1,
        )
        future_multiplier = st.number_input(
            "Future product multiplier",
            value=float(default.future_product_multiplier),
            min_value=0.001,
            step=0.1,
        )
        tasks_json = st.text_area(
            "Quota tasks JSON",
            value=json.dumps(
                [task.model_dump(mode="json") for task in default.quota_tasks],
                ensure_ascii=False,
                indent=2,
            ),
            height=170,
        )
        submitted = st.form_submit_button("Run evaluation")

    if submitted:
        if not run_name.strip():
            st.error("Run name is required")
        else:
            try:
                run_config = RunConfig(
                    reranker_endpoint=reranker_endpoint,
                    global_token_limit=int(global_token_limit),
                    enable_quota_reranking=enable_quota_reranking,
                    task_reranking_temperature=float(task_temp),
                    product_reranking_temperature=float(product_temp),
                    base_product_multiplier=float(base_multiplier),
                    current_product_multiplier=float(current_multiplier),
                    future_product_multiplier=float(future_multiplier),
                    quota_tasks=_tasks_from_json(tasks_json),
                )
                with st.spinner("Running evaluation..."):
                    record, replaced = run_named_evaluation(
                        run_name=run_name.strip(),
                        config=run_config,
                        dataset_path=DATASET_PATH,
                        runs_path=RUNS_PATH,
                    )
                if replaced:
                    st.success(f"Run '{record.run_name}' overwritten")
                else:
                    st.success(f"Run '{record.run_name}' created")
            except Exception as error:
                st.error(f"Failed to run evaluation: {error}")

    records = sorted(load_runs(RUNS_PATH), key=lambda item: item.updated_at)
    col1, col2 = st.columns(2)
    selected_run = None
    with col1:
        event = _records_to_plot(records, "task_kl_div", "product_kl_div", "KL scatter")
        selected_run = _extract_selected_run(event, records) or selected_run
    with col2:
        event = _records_to_plot(records, "task_mse", "product_mse", "MSE scatter")
        selected_run = _extract_selected_run(event, records) or selected_run

    if selected_run is not None:
        for record in records:
            if record.run_name == selected_run:
                _show_run_dialog(record)
                break

    if records:
        chosen_name = st.selectbox(
            "Run details fallback selector",
            options=[record.run_name for record in records],
            index=len(records) - 1,
        )
        for record in records:
            if record.run_name == chosen_name:
                st.json(record.model_dump(mode="json"))
                break


if __name__ == "__main__":
    main()
