# MLEngine

[![PyPI version](https://img.shields.io/pypi/v/ml-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/ml-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/ml-engine-core?logo=python&logoColor=white)](https://pypi.org/project/ml-engine-core/)

MLEngine is a unified, high-performance Machine Learning framework tailored for speed and mathematical transparency. 

Under the hood, MLEngine acts as a pure Python API routing to highly optimized, custom-built C++ native engines (`knn-engine-core` and `nn-engine-core`). By pushing computational bottlenecks—including the entire training loop—down into C++ via `pybind11` and `Eigen`, MLEngine entirely bypasses the Python Global Interpreter Lock (GIL), resulting in execution speeds up to 100x faster than traditional Python-based ML libraries.

## Installation

Install the complete suite via PyPI:

```bash
pip install ml-engine-core
```

## Modules Overview

Once installed, use the unified `mlengine` namespace to access both classical and neural sub-modules.

### 1. `mlengine.knn`
Powered by `knn-engine-core`. Features a highly optimized K-Nearest Neighbors classifier with built-in native Principal Component Analysis (PCA) for dimensionality reduction.

```python
import numpy as np
import mlengine as ml

X_train = np.random.rand(100, 64)
y_train = [str(i % 5) for i in range(100)]

# Configure Engine
cfg = ml.knn.KNNConfig()
cfg.k = 3
cfg.variance = 0.95 # Retain 95% variance via native PCA

engine = ml.knn.KNNEngine(cfg)
engine.train(X_train, y_train, scale=True)

predictions = engine.predict_batch(X_train[:5])
print("KNN Predictions:", predictions)
```

### 2. `mlengine.nn`
Powered by `nn-engine-core`. A multi-layer perceptron framework that executes its full mini-batch gradient descent loop entirely in C++, preventing the Python GIL from interrupting training epochs.

```python
import numpy as np
import mlengine as ml

# Note: MLEngine Neural Nets are hardware-optimized for 32-bit floats
X_train = np.random.rand(100, 4).astype(np.float32)
y_train = np.eye(3)[np.random.choice(3, 100)].astype(np.float32) 
dataloader = ml.nn.DataLoader(X_train, y_train, batch_size=16)

# Build Architecture
class MyModel(ml.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = self.add_module(ml.nn.DenseLayer(4, 16))
        self.relu = self.add_module(ml.nn.ReLULayer())
        self.fc2 = self.add_module(ml.nn.DenseLayer(16, 3))

    def forward(self, tape, x):
        x = self.fc1(tape, x)
        x = self.relu(tape, x)
        return self.fc2(tape, x)

model = MyModel()

# Compile & Train using C++ JIT
optimizer = ml.nn.Adam(learning_rate=0.01)
loss_fn = ml.nn.SoftmaxCrossEntropyLoss()
trainer = ml.nn.JITCompiler(model, optimizer, loss_fn)

# C++ Native Training Loop
trainer.fit(dataloader, epochs=150, verbose=True)

# predict() returns raw logits. Use np.argmax for classification!
logits = model.predict(X_train[:1])
print("Neural Net Logits:", logits)
```

## Architecture & Benchmarks
MLEngine's core philosophy is **Native Loop Hoisting**. Rather than returning to Python after every matrix operation or epoch step, MLEngine passes pointers to C++ once, executes the entire mathematical pipeline using AVX SIMD vectorization and a zero-allocation flat-memory Autograd graph, and returns only when finished.

In standard academic benchmarks (Iris, MNIST Digits, Olivetti Faces), `mlengine.nn` consistently achieves up to a **100x speedup** over Scikit-Learn while maintaining mathematically identical (or superior) accuracy via Log-Sum-Exp fusion.