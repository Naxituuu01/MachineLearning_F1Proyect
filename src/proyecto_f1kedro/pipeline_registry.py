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


def register_pipelines() -> dict[str, Pipeline]:
    """
    Registra los pipelines del proyecto.

    La estructura sigue la metodología CRISP-DM:
    - data_understanding: exploración y comprensión de los datos
    - data_engineering: limpieza, integración y feature engineering
    - data_science: entrenamiento y evaluación de modelos baseline
    """

    data_understanding = du_pipeline()
    data_engineering = de_pipeline()
    data_science = ds_pipeline()

    data_preparation = data_engineering + data_science

    return {
        # Pipelines por fase
        "data_understanding": data_understanding,
        "data_engineering": data_engineering,
        "data_science": data_science,
        "data_preparation": data_preparation,

        # Pipeline completo
        "__default__": (
            data_understanding
            + data_engineering
            + data_science
        ),
    }
