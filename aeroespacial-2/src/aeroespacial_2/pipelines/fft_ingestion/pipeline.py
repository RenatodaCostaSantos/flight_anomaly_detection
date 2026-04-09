"""fft_ingestion pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_fft_flights


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_fft_flights,
                inputs=[
                    "params:fft_ingestion.raw_data_dir",
                    "params:fft_ingestion.imu_source_to_discard",
                    "params:fft_ingestion.flight_keywords",
                    "params:fft_ingestion.fft_relevant_patterns",
                ],
                outputs="fft_ready_flights",
                name="load_fft_flights_node",
            ),
        ]
    )
