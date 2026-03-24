"""feature_engineering pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import engineer_features_for_all_flights


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_features_for_all_flights,
                inputs=[
                    "prepared_flights",
                    "params:feature_engineering.rolling_windows",
                ],
                outputs="feature_engineered_flights",
                name="engineer_features_for_all_flights_node",
            ),
        ]
    )
