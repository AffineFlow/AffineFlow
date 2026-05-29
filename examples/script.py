import argparse
import time
import os
import warnings
import numpy as np

cores = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = cores
os.environ["OMP_NUM_THREADS"] = cores
os.environ["MKL_NUM_THREADS"] = cores

from sklearn.datasets import fetch_olivetti_faces, load_digits, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

import mlengine as ml
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    ml.nn.set_seed(seed)

# ==========================================
# MLEngine Neural Network Architecture
# ==========================================
class BenchmarkNet(ml.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = self.add_module(ml.nn.DenseLayer(input_dim, hidden_dim))
        self.relu = self.add_module(ml.nn.ReLULayer())
        self.fc2 = self.add_module(ml.nn.DenseLayer(hidden_dim, output_dim))

    def forward(self, tape, x):
        x = self.fc1(tape, x)
        x = self.relu(tape, x)
        return self.fc2(tape, x)


def test_dataset(name, X, y, epochs, lr, batch_size, hidden_size, k_neighbors, seed, use_l2=True):
    print(f"\n{'-'*15} Testing {name} {'-'*15}")
    seed_everything(seed)

    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train_full, X_test, y_train_full_oh, y_test_oh = train_test_split(
        X.astype(np.float32), y_onehot.astype(np.float32), test_size=0.2, stratify=y, random_state=seed
    )

    y_train_full = np.argmax(y_train_full_oh, axis=1)
    y_test = np.argmax(y_test_oh, axis=1)

    val_frac = max(0.1, (num_classes + 5) / len(X_train_full))
    
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        X_train_full, y_train_full_oh, test_size=val_frac, stratify=y_train_full, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    X_train_full_scaled = scaler.transform(X_train_full).astype(np.float32)

    # ==========================================
    # 1. MLEngine Neural Network
    # ==========================================
    model = BenchmarkNet(num_features, hidden_size, num_classes)
    optimizer = ml.nn.Adam(learning_rate=np.float32(lr))
    loss_fn = ml.nn.SoftmaxCrossEntropyLoss()
    regularizer = ml.nn.L2Regularizer(l2=np.float32(0.0001)) if use_l2 else None

    train_loader = ml.nn.DataLoader(X_train_scaled, y_train_oh, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = ml.nn.DataLoader(X_val_scaled, y_val_oh, batch_size=batch_size, shuffle=False)
    
    trainer = ml.nn.JITCompiler(model, optimizer, loss_fn, regularizer)

    t0 = time.perf_counter()
    trainer.fit(train_loader, epochs=epochs, val_dataloader=val_loader, tol=1e-4, n_iter_no_change=10, verbose=False)
    t1 = time.perf_counter()

    nn_pred_logits = model.predict(X_test_scaled)
    ml_nn_preds = np.argmax(nn_pred_logits, axis=1)
    
    ml_nn_time = t1 - t0
    ml_nn_acc = accuracy_score(y_test, ml_nn_preds) * 100

    # ==========================================
    # 2. Sklearn Neural Network
    # ==========================================
    sk_nn_model = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=np.float32(lr),
        max_iter=epochs,
        early_stopping=True,       
        validation_fraction=val_frac,
        n_iter_no_change=10,
        tol=1e-4,
        alpha=np.float32(0.0001) if use_l2 else np.float32(0.0),
        random_state=seed,
    )

    t0 = time.perf_counter()
    sk_nn_model.fit(X_train_full_scaled, y_train_full)
    sk_nn_preds = sk_nn_model.predict(X_test_scaled)
    t1 = time.perf_counter()

    sk_nn_time = t1 - t0
    sk_nn_acc = accuracy_score(y_test, sk_nn_preds) * 100

    # ==========================================
    # 3. MLEngine K-Nearest Neighbors
    # ==========================================
    cfg = ml.knn.KNNConfig()
    cfg.k = k_neighbors
    cfg.variance = float(num_features) 
    
    knn_engine = ml.knn.KNNEngine(cfg)
    
    y_train_full_str = [str(label) for label in y_train_full]
    
    t0 = time.perf_counter()
    knn_engine.train(X_train_full_scaled, y_train_full_str, scale=False)
    ml_knn_preds_str = knn_engine.predict_batch(X_test_scaled)
    t1 = time.perf_counter()

    ml_knn_preds = np.array([int(p) for p in ml_knn_preds_str])
    ml_knn_time = t1 - t0
    ml_knn_acc = accuracy_score(y_test, ml_knn_preds) * 100

    # ==========================================
    # 4. Sklearn K-Nearest Neighbors
    # ==========================================
    sk_knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
    
    t0 = time.perf_counter()
    sk_knn_model.fit(X_train_full_scaled, y_train_full)
    sk_knn_preds = sk_knn_model.predict(X_test_scaled)
    t1 = time.perf_counter()

    sk_knn_time = t1 - t0
    sk_knn_acc = accuracy_score(y_test, sk_knn_preds) * 100

    # ==========================================
    # Results Display
    # ==========================================
    print("\n[ Neural Networks (MLP) ]")
    print(f"🚀 MLEngine (JIT C++) — Accuracy: {ml_nn_acc:6.2f}%  |  Time: {ml_nn_time:.4f}s")
    print(f"🐢 Sklearn (Adam)   — Accuracy: {sk_nn_acc:6.2f}%  |  Time: {sk_nn_time:.4f}s")
    print(f"⚡ NN Speedup       — {sk_nn_time / ml_nn_time:.2f}x Faster")

    print("\n[ K-Nearest Neighbors ]")
    print(f"🚀 MLEngine (Native)  — Accuracy: {ml_knn_acc:6.2f}%  |  Time: {ml_knn_time:.4f}s")
    print(f"🐢 Sklearn (KDTree) — Accuracy: {sk_knn_acc:6.2f}%  |  Time: {sk_knn_time:.4f}s")
    print(f"⚡ KNN Speedup      — {sk_knn_time / ml_knn_time:.2f}x Faster")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLEngine complete benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("🔥 MLEngine Full Suite Benchmark Starting...\n")

    iris = load_iris()
    test_dataset(
        name="Iris Flower", X=iris.data, y=iris.target, 
        epochs=100, lr=0.01, batch_size=16, hidden_size=16, k_neighbors=3, seed=args.seed
    )

    digits = load_digits()
    test_dataset(
        name="Handwritten Digits", X=digits.data, y=digits.target, 
        epochs=100, lr=0.001, batch_size=32, hidden_size=128, k_neighbors=5, seed=args.seed + 1
    )

    faces = fetch_olivetti_faces()
    test_dataset(
        name="Olivetti Faces", X=faces.data, y=faces.target, 
        epochs=100, lr=0.001, batch_size=32, hidden_size=128, k_neighbors=3, seed=args.seed + 2
    )

if __name__ == "__main__":
    main()