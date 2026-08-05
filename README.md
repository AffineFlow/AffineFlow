# AffineFlow

[![PyPI version](https://img.shields.io/pypi/v/AffineFlow?logo=pypi&logoColor=white)](https://pypi.org/project/AffineFlow/)
[![Python](https://img.shields.io/pypi/pyversions/AffineFlow?logo=python&logoColor=white)](https://pypi.org/project/AffineFlow/)

AffineFlow is a unified, high-performance Machine Learning framework tailored for speed and mathematical transparency. 

Under the hood, AffineFlow acts as a pure Python API routing to highly optimized, custom-built C++ native engines (`affineflow-ml` and `affineflow-nn`). By pushing computational bottlenecks—including the entire training loop—down into C++ via `pybind11` and `Eigen`, AffineFlow entirely bypasses the Python Global Interpreter Lock (GIL), resulting in execution speeds up to 100x faster than traditional Python-based ML libraries.

## Installation

Install the complete suite via PyPI:

```bash
pip install affineflow
```

## Modules Overview

Once installed, use the unified `affineflow` namespace to access both classical and neural sub-modules.

### 1. `affineflow.ml`
Powered by `affineflow-ml`. Features a highly optimized K-Nearest Neighbors classifier and Principal Component Analysis (PCA) adhering strictly to Scikit-Learn API standards.

```python
import numpy as np
import affineflow as af

X_train = np.random.rand(100, 64)
y_train = ["A" if i % 2 == 0 else "B" for i in range(100)]

# Initialize and fit model using Scikit-Learn standards
knn = af.ml.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_train[:5])
print("KNN Predictions:", predictions)
```

### 2. `affineflow.nn`
Powered by `affineflow-nn`. A neural network framework that executes its full mini-batch gradient descent loop entirely in C++, preventing the Python GIL from interrupting training epochs.

```python
import numpy as np
import affineflow_nn as nn

X_train = np.random.rand(100, 1, 64, 64).astype(np.float32)
y_train = np.eye(40, dtype=np.float32)[np.random.choice(40, 100)]

class AffineFlowNNDeepCNN(nn.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2dLayer(1, 16, in_h, in_w, kernel_size=5, stride=1, pad=2)
        self.act1 = nn.LeakyReLULayer(0.01)
        self.pool1 = nn.MaxPool2dLayer(16, in_h, in_w, kernel_size=2, stride=2, pad=0)
        
        out_h1, out_w1 = in_h // 2, in_w // 2
        self.conv2 = nn.Conv2dLayer(16, 32, out_h1, out_w1, kernel_size=3, stride=1, pad=1)
        self.act2 = nn.LeakyReLULayer(0.01)
        self.pool2 = nn.MaxPool2dLayer(32, out_h1, out_w1, kernel_size=2, stride=2, pad=0)
        
        out_h2, out_w2 = out_h1 // 2, out_w1 // 2
        self.flatten = nn.FlattenLayer()
        self.fc = nn.DenseLayer(32 * out_h2 * out_w2, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

model = AffineFlowNNDeepCNN(64, 64, 40)
optimizer = nn.Adam(learning_rate=0.001)
optimizer.set_parameters(model.parameters())
loss_fn = nn.SoftmaxCrossEntropyLoss()

trainer = nn.JITCompiler(model, optimizer, loss_fn)
dataloader = nn.DataLoader(X_train, y_train, batch_size=32, shuffle=True, drop_last=True)
trainer.fit(dataloader, epochs=40, verbose=False)
trainer.save_checkpoint("faces_model")
```

## Architecture & Benchmarks
AffineFlow's core philosophy is **Native Loop Hoisting**. Rather than returning to Python after every matrix operation or epoch step, AffineFlow passes pointers to C++ once, executes the entire mathematical pipeline using AVX SIMD vectorization and a zero-allocation flat-memory Autograd graph, and returns only when finished.