from kedro.pipeline import Pipeline, node
from .nodes import drop_columns, fill_missing_values, create_features, save_clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=drop_columns,
                inputs=["races", "parameters"],
                outputs="races_dropped",
                name="drop_columns_races"
            ),
            node(
                func=fill_missing_values,
                inputs="races_dropped",
                outputs="races_filled",
                name="fill_missing_races"
            ),
            node(
                func=create_features,
                inputs=["races_filled", "drivers", "results"],
                outputs="f1_features",
                name="create_features"
            ),
            node(
                func=save_clean_data,
                inputs=["f1_features", "parameters"],
                outputs="clean_f1_data",
                name="save_clean_data"
            )
        ]
    )
