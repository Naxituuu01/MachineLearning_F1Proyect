from kedro.pipeline import Pipeline, node
from .nodes import integrate_datasets, build_features, select_model_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=integrate_datasets,
                inputs=[
                    "results_raw",
                    "races_raw",
                    "drivers_raw",
                    "constructors_raw",
                    "circuits_raw",
                ],
                outputs="f1_integrated",
                name="integrate_datasets_node",
            ),
            node(
                func=build_features,
                inputs="f1_integrated",
                outputs="f1_features",
                name="feature_engineering_node",
            ),
            node(
                func=select_model_features,
                inputs="f1_features",
                outputs="clean_f1_data",
                name="select_features_node",
            ),
        ]
    )
