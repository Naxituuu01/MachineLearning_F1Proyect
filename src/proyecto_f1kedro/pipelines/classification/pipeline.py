from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_and_evaluate_classification

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_and_evaluate_classification,
                inputs=dict(
                    model_input_classification="model_input_classification",
                    modeling="params:modeling",
                ),
                outputs=[
                    "classification_metrics_table",
                    "classification_metrics_summary",
                    "classification_metrics_plot",
                    "best_model_classification",
                ],
                name="train_and_evaluate_classification_node",
            )
        ]
    )
