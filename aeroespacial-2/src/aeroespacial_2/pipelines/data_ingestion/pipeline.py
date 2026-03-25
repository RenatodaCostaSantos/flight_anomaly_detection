"""data_ingestion pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_all_flights


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_all_flights,
                inputs=[
                    "params:data_ingestion.raw_data_dir",
                    "params:data_ingestion.imu_source_to_discard",
                    "params:data_ingestion.flight_keywords",
                ],
                outputs="preprocessed_flights",
                name="load_all_flights_node",
            ),
        ]
    )
