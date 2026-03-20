"""model_training pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_training_data, evaluate_model, train_isolation_forest


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_training_data,
                inputs=[
                    "prepared_flights",
                    "params:model_training.window_size",
                    "params:model_training.n_top_features",
                    "params:model_training.train_ratio",
                    "params:model_training.target_col",
                    "params:model_training.timestamp_col",
                ],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "test_timestamps",
                    "selected_features",
                ],
                name="build_training_data_node",
            ),
            node(
                func=train_isolation_forest,
                inputs=[
                    "X_train",
                    "params:model_training.contamination",
                    "params:model_training.n_estimators",
                ],
                outputs="isolation_forest",
                name="train_isolation_forest_node",
            ),
            node(
                func=evaluate_model,
                inputs=["isolation_forest", "X_test", "y_test", "test_timestamps"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
