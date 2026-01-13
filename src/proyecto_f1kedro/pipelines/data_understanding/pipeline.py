from kedro.pipeline import Pipeline, node
from .nodes import load_dataset, summary_statistics, check_missing_values, plot_distributions

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_dataset,
                inputs="races_raw",
                outputs="races_loaded",
                name="load_races"
            ),
            node(
                func=summary_statistics,
                inputs="races_loaded",
                outputs="races_summary",
                name="summary_races"
            ),
            node(
                func=check_missing_values,
                inputs="races_loaded",
                outputs="races_missing",
                name="missing_races"
            ),
            node(
                func=plot_distributions,
                inputs="races_loaded",
                outputs=None,
                name="plot_races"
            )
        ]
    )
