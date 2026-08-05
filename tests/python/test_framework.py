import pytest
import numpy as np

import affineflow as af
import affineflow.nn as nn
import affineflow.ml as ml
from affineflow.adapters import NNEstimator
from affineflow.compose import Pipeline

def test_lazy_loading_and_dynamic_all():
    """Verify that the root __getattr__ and __dir__ correctly expose subpackages."""
    # Test __dir__ (autocomplete support)
    af_dir = dir(af)
    assert "ml" in af_dir
    assert "nn" in af_dir
    assert "compose" in af_dir
    assert "adapters" in af_dir

    # Test dynamic imports from __all__ equivalents in subpackages
    import affineflow.compose as compose
    import affineflow.adapters as adapters
    
    assert "Pipeline" in dir(compose)
    assert "NNEstimator" in dir(adapters)

def test_nn_estimator_adapter():
    """Verify that the NNEstimator correctly wraps native C++ engine components."""
    nn.set_seed(42)
    
    # Generate synthetic regression data
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.rand(100, 1).astype(np.float32)

    # Build Native Components
    model = nn.DenseLayer(10, 1)
    optimizer = nn.Adam(learning_rate=0.01)
    optimizer.set_parameters(model.parameters())
    loss_fn = nn.MSELoss()

    # Wrap in Adapter
    estimator = NNEstimator(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=5,
        batch_size=16,
        verbose=False
    )

    # Test Scikit-Learn standard API
    estimator.fit(X, y)
    preds = estimator.predict(X[:5])
    
    assert preds.shape == (5, 1), "Estimator output shape mismatch"
    assert preds.dtype == np.float32, "Estimator output dtype mismatch"
    assert not np.isnan(preds).any(), "Estimator output contains NaNs"

def test_hybrid_pipeline_ml_to_nn():
    """
    Verify that AffineFlow-ML transformers (Eigen/double) seamlessly hand off 
    data to AffineFlow-NN components (FlatStorage/float) via the Pipeline.
    """
    nn.set_seed(42)
    
    # Generate data (float64 for ML processing)
    X = np.random.rand(200, 50).astype(np.float64) 
    y = np.random.rand(200, 1).astype(np.float32)

    # 1. Native ML Transformer (reduces from 50 to 5 features)
    pca = ml.PCA(n_components=5.0) 

    # 2. Native NN Estimator (expects the 5 features from PCA)
    model = nn.DenseLayer(5, 1)
    optimizer = nn.SGD(learning_rate=0.05)
    optimizer.set_parameters(model.parameters())
    
    estimator = NNEstimator(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        epochs=3,
        batch_size=32,
        verbose=False
    )

    # 3. Build and execute the Pipeline
    pipeline = Pipeline([
        ("pca", pca),
        ("nn", estimator)
    ])

    pipeline.fit(X, y)
    preds = pipeline.predict(X[:10])
    
    assert preds.shape == (10, 1), "Pipeline output shape mismatch"
    assert preds.dtype == np.float32, "Pipeline output dtype mismatch"