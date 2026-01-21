from kedro.pipeline import Pipeline, node, pipeline
from .nodes import build_model_inputs_from_raw


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_model_inputs_from_raw,
                inputs=dict(
                    results_raw="results_raw",
                    races_raw="races_raw",
                    targets="params:targets",
                    data_prep="params:data_preparation",
                ),
                outputs=["model_input_classification", "model_input_regression"],
                name="build_model_inputs_from_raw_node",
            )
        ]
    )
