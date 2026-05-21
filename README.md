# MLEngine

[![PyPI version](https://img.shields.io/pypi/v/ml-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/ml-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/ml-engine-core?logo=python&logoColor=white)](https://pypi.org/project/ml-engine-core/)

MLEngine is a unified, high-performance Machine Learning framework tailored for speed and mathematical transparency. 

Under the hood, MLEngine acts as a pure Python API routing to highly optimized, custom-built C++ native engines (`knn-engine-core` and `nn-engine-core`). By pushing computational bottlenecks—including the entire training loop—down into C++ via `pybind11` and `Eigen`, MLEngine entirely bypasses the Python Global Interpreter Lock (GIL), resulting in execution speeds up to 30x faster than traditional Python-based ML libraries.

## Installation

Install the complete suite via PyPI:

```bash
pip install ml-engine-core
```

## Modules Overview

Once installed, use the unified `ml_core` namespace to access both classical and neural sub-modules.

### 1. `ml_core.classical`
Powered by `knn-engine-core`. Features a highly optimized K-Nearest Neighbors classifier with built-in native Principal Component Analysis (PCA) for dimensionality reduction.

```python
import numpy as np
from ml_core.classical import KNNEngine, KNNConfig

X_train = np.random.rand(100, 64)
y_train = [str(i % 5) for i in range(100)]

# Configure Engine
cfg = KNNConfig()
cfg.k = 3
cfg.variance = 0.95 # Retain 95% variance via native PCA

engine = KNNEngine(cfg)
engine.train(X_train, y_train, scale=True)

predictions = engine.predict_batch(X_train[:5])
print("KNN Predictions:", predictions)
```

### 2. `ml_core.neural`
Powered by `nn-engine-core`. A multi-layer perceptron framework that executes its full mini-batch gradient descent loop entirely in C++, preventing the Python GIL from interrupting training epochs.

```python
import numpy as np
from ml_core.neural import Model, DenseLayer, ReLULayer, SoftmaxLayer, CategoricalCrossEntropyLoss

X_train = np.random.rand(100, 4).astype(np.float64)
y_train = np.eye(3)[np.random.choice(3, 100)].astype(np.float64) 

# Build Architecture
model = Model()
model.add(DenseLayer(4, 16))
model.add(ReLULayer())
model.add(DenseLayer(16, 3))
model.add(SoftmaxLayer())

model.compile(CategoricalCrossEntropyLoss())

# C++ Native Training Loop
model.fit(X_train, y_train, epochs=150, learning_rate=0.05, batch_size=16)

probs = model.predict(X_train[:1])
print("Neural Net Probs:", probs)
```

## Architecture & Benchmarks
MLEngine's core philosophy is **Native Loop Hoisting**. Rather than returning to Python after every matrix operation or epoch step, MLEngine passes pointers to C++ once, executes the entire mathematical pipeline using Eigen's `noalias()` buffers, and returns only when finished.

In standard academic benchmarks (Iris, MNIST Digits, Olivetti Faces), `ml_core` consistently achieves 2x to 30x speedups over Scikit-Learn while maintaining equal or superior accuracy.
