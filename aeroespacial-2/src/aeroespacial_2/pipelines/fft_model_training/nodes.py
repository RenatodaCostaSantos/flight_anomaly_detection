"""Nodes for the fft_model_training pipeline.

Reuses the same node functions as model_training — only the input dataset
and catalog outputs differ (FFT-derived features vs. time-domain features).
"""
from aeroespacial_2.pipelines.model_training.nodes import (  # noqa: F401
    build_training_data,
    create_windows,
    evaluate_model,
    fit_and_scale,
    select_top_features_effect_size,
    split_windows,
    train_isolation_forest,
)
