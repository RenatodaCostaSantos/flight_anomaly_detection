"""data_preparation pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_all_flights


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_all_flights,
                inputs=[
                    "preprocessed_flights",
                    "params:data_preparation.cut_seconds",
                    "params:data_preparation.min_std_threshold",
                    "params:data_preparation.detrend_seconds",
                ],
                outputs="prepared_flights",
                name="prepare_all_flights_node",
            ),
        ]
    )
