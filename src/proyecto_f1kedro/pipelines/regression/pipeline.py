from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_and_evaluate_regression

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_and_evaluate_regression,
                inputs=dict(
                    model_input_regression="model_input_regression_with_clusters",
                    modeling="params:modeling",
                ),
                outputs=[
                    "regression_metrics_table",
                    "regression_metrics_summary",
                    "regression_metrics_plot",
                    "best_model_regression",
                    "regression_predictions_test",
                    "regression_feature_importances",
                ],
                name="train_and_evaluate_regression_node",
            ),
        ]
    )
