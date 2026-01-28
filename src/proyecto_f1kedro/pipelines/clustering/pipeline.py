from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_race_level_features,
    fit_clustering,
    attach_race_clusters_to_model_input,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_race_level_features,
                inputs=dict(
                    model_input_regression="model_input_regression",
                    params="params:clustering",
                ),
                outputs="clustering_race_features",
                name="build_race_level_features_node",
            ),
            node(
                func=fit_clustering,
                inputs=dict(
                    clustering_race_features="clustering_race_features",
                    params="params:clustering",
                ),
                outputs=[
                    "clustering_labels_race",
                    "clustering_metrics",
                    "clustering_plot_elbow",
                    "clustering_plot_silhouette",
                    "clustering_bundle",
                    "clustering_cluster_profile",
                    "clustering_plot_pca2d",
                ],
                name="fit_clustering_node",
            ),
            node(
                func=attach_race_clusters_to_model_input,
                inputs=dict(
                    model_input="model_input_regression",
                    clustering_labels_race="clustering_labels_race",
                    params="params:clustering",
                ),
                outputs="model_input_regression_with_clusters",
                name="attach_clusters_to_regression_model_input_node",
            ),
            node(
                func=attach_race_clusters_to_model_input,
                inputs=dict(
                    model_input="model_input_classification",
                    clustering_labels_race="clustering_labels_race",
                    params="params:clustering",
                ),
                outputs="model_input_classification_with_clusters",
                name="attach_clusters_to_classification_model_input_node",
            ),
        ]
    )