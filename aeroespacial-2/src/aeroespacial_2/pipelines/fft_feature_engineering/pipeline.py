"""fft_feature_engineering pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import engineer_fft_features_for_all_flights


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_fft_features_for_all_flights,
                inputs=[
                    "fft_prepared_flights",
                    "params:fft_feature_engineering.rolling_windows",
                    "params:fft_feature_engineering.fft_windows",
                ],
                outputs="fft_feature_engineered_flights",
                name="engineer_fft_features_for_all_flights_node",
            ),
        ]
    )
