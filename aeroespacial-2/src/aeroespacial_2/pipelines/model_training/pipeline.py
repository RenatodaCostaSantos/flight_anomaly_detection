"""model_training pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_training_data, evaluate_model, fit_and_scale, train_isolation_forest


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
                func=fit_and_scale,
                inputs=["X_train", "X_test"],
                outputs=["X_train_scaled", "X_test_scaled", "feature_scaler"],
                name="fit_and_scale_node",
            ),
            node(
                func=train_isolation_forest,
                inputs=[
                    "X_train_scaled",
                    "params:model_training.contamination",
                    "params:model_training.n_estimators",
                ],
                outputs="isolation_forest",
                name="train_isolation_forest_node",
            ),
            node(
                func=evaluate_model,
                inputs=["isolation_forest", "X_test_scaled", "y_test", "test_timestamps"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
