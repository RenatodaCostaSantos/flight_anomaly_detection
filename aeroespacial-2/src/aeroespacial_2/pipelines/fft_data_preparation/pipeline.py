"""fft_data_preparation pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_all_fft_flights


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_all_fft_flights,
                inputs=[
                    "fft_ready_flights",
                    "params:fft_data_preparation.cut_seconds",
                    "params:fft_data_preparation.min_std_threshold",
                    "params:fft_data_preparation.detrend_seconds",
                ],
                outputs="fft_prepared_flights",
                name="prepare_all_fft_flights_node",
            ),
        ]
    )
