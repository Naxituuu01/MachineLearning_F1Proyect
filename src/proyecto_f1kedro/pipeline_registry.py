from kedro.pipeline import Pipeline
from kedro.pipeline import pipeline as kedro_pipeline

from proyecto_f1kedro.pipelines.data_understanding import create_pipeline as du_pipeline
from proyecto_f1kedro.pipelines.data_engineering import create_pipeline as de_pipeline
from proyecto_f1kedro.pipelines.data_science import create_pipeline as ds_pipeline
from proyecto_f1kedro.pipelines import model_input as model_input_pipeline
from proyecto_f1kedro.pipelines.classification import create_pipeline as cls_pipeline
from proyecto_f1kedro.pipelines.regression import create_pipeline as reg_pipeline
from proyecto_f1kedro.pipelines.clustering import create_pipeline as clustering_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    data_understanding = du_pipeline()
    data_engineering = de_pipeline()
    model_input = model_input_pipeline.create_pipeline()
    data_science = ds_pipeline()

    clustering = clustering_pipeline()
    classification = cls_pipeline()
    regression = reg_pipeline()

    # legacy
    data_preparation = data_engineering + model_input + data_science

    # training con clustering entre medio
    training = model_input + clustering + classification + regression

    full = data_understanding + data_engineering + training

    return {
        "data_understanding": data_understanding,
        "data_engineering": data_engineering,
        "model_input": model_input,
        "clustering": clustering,
        "classification": classification,
        "regression": regression,
        "training": training,
        "full": full,
        "data_science": data_science,
        "data_preparation": data_preparation,
        "__default__": training,
    }
