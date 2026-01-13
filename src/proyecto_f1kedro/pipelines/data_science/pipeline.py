from kedro.pipeline import Pipeline, node
from .nodes import train_classification_model, train_regression_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_classification_model,
                inputs="clean_f1_data",
                outputs="classification_metrics",
                name="classification_node",
            ),
            node(
                func=train_regression_model,
                inputs="clean_f1_data",
                outputs="regression_metrics",
                name="regression_node",
            ),
        ]
    )
