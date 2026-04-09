"""fft_model_training pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_training_data, evaluate_model, fit_and_scale, train_isolation_forest


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_training_data,
                inputs=[
                    "fft_feature_engineered_flights",
                    "params:fft_model_training.window_size",
                    "params:fft_model_training.n_top_features",
                    "params:fft_model_training.train_ratio",
                    "params:fft_model_training.target_col",
                    "params:fft_model_training.timestamp_col",
                    "params:fft_model_training.skip_seconds",
                ],
                outputs=[
                    "fft_X_train",
                    "fft_X_test",
                    "fft_y_train",
                    "fft_y_test",
                    "fft_test_timestamps",
                    "fft_selected_features",
                ],
                name="fft_build_training_data_node",
            ),
            node(
                func=fit_and_scale,
                inputs=["fft_X_train", "fft_X_test"],
                outputs=["fft_X_train_scaled", "fft_X_test_scaled", "fft_feature_scaler"],
                name="fft_fit_and_scale_node",
            ),
            node(
                func=train_isolation_forest,
                inputs=[
                    "fft_X_train_scaled",
                    "params:fft_model_training.contamination",
                    "params:fft_model_training.n_estimators",
                    "params:fft_model_training.max_samples",
                    "params:fft_model_training.max_features",
                ],
                outputs="fft_isolation_forest",
                name="fft_train_isolation_forest_node",
            ),
            node(
                func=evaluate_model,
                inputs=["fft_isolation_forest", "fft_X_test_scaled", "fft_y_test", "fft_test_timestamps"],
                outputs="fft_model_metrics",
                name="fft_evaluate_model_node",
            ),
        ]
    )
