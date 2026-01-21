from kedro.pipeline import Pipeline

from proyecto_f1kedro.pipelines.data_understanding import (
    create_pipeline as du_pipeline,
)
from proyecto_f1kedro.pipelines.data_engineering import (
    create_pipeline as de_pipeline,
)
from proyecto_f1kedro.pipelines.data_science import (
    create_pipeline as ds_pipeline,
)
from proyecto_f1kedro.pipelines import model_input as model_input_pipeline

from proyecto_f1kedro.pipelines.classification import (
    create_pipeline as cls_pipeline,
)
from proyecto_f1kedro.pipelines.regression import (
    create_pipeline as reg_pipeline,
)



def register_pipelines() -> dict[str, Pipeline]:
    """
    Registra los pipelines del proyecto.

    La estructura sigue la metodología CRISP-DM:
    - data_understanding: exploración y comprensión de los datos
    - data_engineering: limpieza, integración y feature engineering
    - model_input: construcción de datasets finales para ML (clasificación y regresión)
    - data_science: entrenamiento y evaluación de modelos (baseline actual)
    """

    data_understanding = du_pipeline()
    data_engineering = de_pipeline()
    model_input = model_input_pipeline.create_pipeline()
    data_science = ds_pipeline()
    classification = cls_pipeline()
    regression = reg_pipeline()



    # Si quieres mantener "data_preparation" como un "shortcut" útil:
    data_preparation = data_engineering + model_input + data_science

    return {
        "data_understanding": data_understanding,
        "data_engineering": data_engineering,
        "model_input": model_input,
        "data_science": data_science,
        "data_preparation": data_preparation,
        "classification": classification,
        "regression": regression,



        "__default__": (
            data_understanding
            + data_engineering
            + model_input
            + data_science
        ),
    }
