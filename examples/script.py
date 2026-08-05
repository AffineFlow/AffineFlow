import time
import warnings
import numpy as np

# Suppress Scikit-Learn convergence warnings for a clean CLI output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

device = torch.device("cpu")

from sklearn.datasets import fetch_california_housing, fetch_olivetti_faces
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier

import affineflow.ml as ml
import affineflow.nn as afnn

SEEDS = [42, 1337, 2026]


# ==========================================
# TASK 1: KNN CLASSIFICATION (OLIVETTI FACES)
# scikit-learn vs affineflow.ml
# ==========================================
def run_knn_classification(seeds=SEEDS):
    print(f"\n{'-'*20} BENCHMARK: KNN CLASSIFICATION (OLIVETTI FACES) {'-'*20}")

    faces = fetch_olivetti_faces()
    X, y = faces.data.astype(np.float64), faces.target

    # KNeighborsClassifier has no internal randomness on either side, so the
    # seed varies the train/test split instead -- that's what actually moves
    # accuracy on a dataset this small (80 test images across 40 classes).

    # --- 1. Scikit-Learn ---
    print("Evaluating Scikit-Learn KNeighborsClassifier across seeds...")
    best_sk_acc = -1.0
    best_sk_results = None

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        sk_clf = SkKNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")

        t0 = time.perf_counter()
        sk_clf.fit(X_train, y_train)
        sk_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        sk_preds = sk_clf.predict(X_test)
        sk_infer_time = time.perf_counter() - t0

        sk_acc = accuracy_score(y_test, sk_preds) * 100
        if sk_acc > best_sk_acc:
            best_sk_acc = sk_acc
            best_sk_results = (sk_acc, sk_train_time, sk_infer_time)
    sk_acc, sk_train_time, sk_infer_time = best_sk_results

    # --- 2. AffineFlow-ML ---
    print("Evaluating AffineFlow-ML KNeighborsClassifier across seeds...")
    best_af_acc = -1.0
    best_af_results = None

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        af_clf = ml.KNeighborsClassifier(
            n_neighbors=5,
            metric=ml.DistanceMetric.COSINE,
            weights=ml.Weights.DISTANCE,
        )

        t0 = time.perf_counter()
        af_clf.fit(X_train, [str(label) for label in y_train])
        af_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        af_preds = np.array(af_clf.predict(X_test), dtype=int)
        af_infer_time = time.perf_counter() - t0

        af_acc = accuracy_score(y_test, af_preds) * 100
        if af_acc > best_af_acc:
            best_af_acc = af_acc
            best_af_results = (af_acc, af_train_time, af_infer_time)
    af_acc, af_train_time, af_infer_time = best_af_results

    # Results
    print(f"\n--- KNN Results (Best over {len(seeds)} Seeds) ---")
    print(f"Scikit-Learn  | Acc: {sk_acc:.2f}% | Train: {sk_train_time:.4f}s | Infer: {sk_infer_time:.4f}s")
    print(f"AffineFlow-ML | Acc: {af_acc:.2f}% | Train: {af_train_time:.4f}s | Infer: {af_infer_time:.4f}s")


# ==========================================
# TASK 2: DEEP REGRESSION (CALIFORNIA HOUSING)
# scikit-learn vs PyTorch vs AffineFlow-NN (JIT) vs AffineFlow-NN (Eager)
# ==========================================
class PyTorchDeepMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class AffineFlowNNDeepMLP(afnn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = afnn.DenseLayer(in_features, 128)
        self.act1 = afnn.ReLULayer()

        self.fc2 = afnn.DenseLayer(128, 64)
        self.act2 = afnn.ReLULayer()

        self.fc3 = afnn.DenseLayer(64, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def run_regression_benchmark(seeds=SEEDS):
    print(f"\n{'-'*20} BENCHMARK: DEEP REGRESSION (CALIFORNIA HOUSING) {'-'*20}")

    X, y = fetch_california_housing(return_X_y=True)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y.astype(np.float32), test_size=0.2, random_state=42
    )

    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler_x.transform(X_test).astype(np.float32)

    epochs = 50
    batch_size = 128
    lr = 0.01
    weight_decay = 1e-4

    # --- 1. Scikit-Learn ---
    print("Evaluating Scikit-Learn MLPRegressor across seeds...")
    best_sk_mse = float("inf")
    best_sk_results = None

    for seed in seeds:
        sk_model = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
            alpha=weight_decay, batch_size=batch_size, learning_rate_init=lr,
            max_iter=epochs, random_state=seed, early_stopping=False
        )
        t0 = time.perf_counter()
        sk_model.fit(X_train_s, y_train.ravel())
        sk_time = time.perf_counter() - t0
        sk_preds = sk_model.predict(X_test_s)
        sk_mse = mean_squared_error(y_test, sk_preds)
        sk_r2 = r2_score(y_test, sk_preds)

        if sk_mse < best_sk_mse:
            best_sk_mse = sk_mse
            best_sk_results = (sk_mse, sk_r2, sk_time)
    sk_mse, sk_r2, sk_time = best_sk_results

    # --- 2. PyTorch ---
    print("Evaluating PyTorch (ATen) across seeds...")
    best_pt_mse = float("inf")
    best_pt_results = None

    for seed in seeds:
        torch.manual_seed(seed)
        pt_model = PyTorchDeepMLP(X_train.shape[1]).to(device)
        pt_opt = optim.Adam(pt_model.parameters(), lr=lr, weight_decay=weight_decay)
        pt_loss_fn = nn.MSELoss()

        train_ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train))
        train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        t0 = time.perf_counter()
        pt_model.train()
        for epoch in range(epochs):
            for bx, by in train_loader:
                pt_opt.zero_grad()
                loss = pt_loss_fn(pt_model(bx), by)
                loss.backward()
                pt_opt.step()
        pt_time = time.perf_counter() - t0

        pt_model.eval()
        with torch.no_grad():
            pt_preds = pt_model(torch.tensor(X_test_s)).numpy()
        pt_mse = mean_squared_error(y_test, pt_preds)
        pt_r2 = r2_score(y_test, pt_preds)

        if pt_mse < best_pt_mse:
            best_pt_mse = pt_mse
            best_pt_results = (pt_mse, pt_r2, pt_time)
    pt_mse, pt_r2, pt_time = best_pt_results

    # --- 3. AffineFlow-NN JIT ---
    print("Evaluating AffineFlow-NN (JIT) across seeds...")
    best_nn_mse = float("inf")
    best_nn_results = None

    for seed in seeds:
        afnn.set_seed(seed)
        nn_model = AffineFlowNNDeepMLP(X_train.shape[1])
        nn_opt = afnn.Adam(learning_rate=lr)
        nn_loss = afnn.MSELoss()
        nn_reg = afnn.L2Regularizer(l2=weight_decay)

        trainer = afnn.JITCompiler(nn_model, nn_opt, nn_loss, regularizer=nn_reg)
        nn_loader = afnn.DataLoader(X_train_s, y_train, batch_size=batch_size, shuffle=True, drop_last=True)

        t0 = time.perf_counter()
        trainer.fit(nn_loader, epochs=epochs, verbose=False)
        nn_time = time.perf_counter() - t0

        nn_preds = np.array(nn_model.predict(X_test_s))
        nn_mse = mean_squared_error(y_test, nn_preds)
        nn_r2 = r2_score(y_test, nn_preds)

        if nn_mse < best_nn_mse:
            best_nn_mse = nn_mse
            best_nn_results = (nn_mse, nn_r2, nn_time)
    nn_mse, nn_r2, nn_time = best_nn_results

    # --- 4. AffineFlow-NN Eager (PyTorch-like Implicit Execution) ---
    print("Evaluating AffineFlow-NN (Eager) across seeds...")
    best_eag_mse = float("inf")
    best_eag_results = None

    for seed in seeds:
        afnn.set_seed(seed)
        eag_model = AffineFlowNNDeepMLP(X_train.shape[1])
        eag_opt = afnn.Adam(learning_rate=lr)
        eag_opt.set_parameters(eag_model.parameters())
        eag_loss = afnn.MSELoss()
        eag_reg = afnn.L2Regularizer(l2=weight_decay)

        eag_loader = afnn.DataLoader(X_train_s, y_train, batch_size=batch_size, shuffle=True, drop_last=True)

        bx = afnn.Tensor(np.zeros((batch_size, X_train.shape[1]), dtype=np.float32))
        by = afnn.Tensor(np.zeros((batch_size, 1), dtype=np.float32))

        t0 = time.perf_counter()
        eag_model.train(True)
        for epoch in range(epochs):
            eag_loader.reset()
            while eag_loader.has_next():
                eag_loader.next_batch(bx, by)

                eag_opt.zero_grad()
                preds = eag_model(bx)
                loss = eag_loss.forward(preds, by)
                eag_loss.backward()
                eag_reg.apply(eag_model.parameters())
                eag_opt.step()

        eag_time = time.perf_counter() - t0

        eag_preds = np.array(eag_model.predict(X_test_s))
        eag_mse = mean_squared_error(y_test, eag_preds)
        eag_r2 = r2_score(y_test, eag_preds)

        if eag_mse < best_eag_mse:
            best_eag_mse = eag_mse
            best_eag_results = (eag_mse, eag_r2, eag_time)
    eag_mse, eag_r2, eag_time = best_eag_results

    # Results
    print(f"\n--- Regression Results (Best over {len(seeds)} Seeds) ---")
    print(f"Scikit-Learn      | MSE: {sk_mse:.4f} | R2: {sk_r2:.4f} | Time: {sk_time:.4f}s")
    print(f"PyTorch           | MSE: {pt_mse:.4f} | R2: {pt_r2:.4f} | Time: {pt_time:.4f}s")
    print(f"AffineFlow-NN JIT | MSE: {nn_mse:.4f} | R2: {nn_r2:.4f} | Time: {nn_time:.4f}s")
    print(f"AffineFlow-NN Eag | MSE: {eag_mse:.4f} | R2: {eag_r2:.4f} | Time: {eag_time:.4f}s")


# ==========================================
# TASK 3: DEEP SPATIAL CLASSIFICATION (OLIVETTI FACES)
# PyTorch vs AffineFlow-NN (JIT) vs AffineFlow-NN (Eager)
# (scikit-learn has no first-class CNN, so it's excluded here)
# ==========================================
class PyTorchDeepCNN(nn.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.in_h, self.in_w = in_h, in_w

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.LeakyReLU(0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU(0.01)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * (in_h // 4) * (in_w // 4), num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.in_h, self.in_w)
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


class AffineFlowNNDeepCNN(afnn.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.conv1 = afnn.Conv2dLayer(1, 16, in_h, in_w, kernel_size=5, stride=1, pad=2)
        self.act1 = afnn.LeakyReLULayer(0.01)
        self.pool1 = afnn.MaxPool2dLayer(16, in_h, in_w, kernel_size=2, stride=2, pad=0)

        out_h1, out_w1 = in_h // 2, in_w // 2
        self.conv2 = afnn.Conv2dLayer(16, 32, out_h1, out_w1, kernel_size=3, stride=1, pad=1)
        self.act2 = afnn.LeakyReLULayer(0.01)
        self.pool2 = afnn.MaxPool2dLayer(32, out_h1, out_w1, kernel_size=2, stride=2, pad=0)

        out_h2, out_w2 = out_h1 // 2, out_w1 // 2

        self.flatten = afnn.FlattenLayer()
        self.fc = afnn.DenseLayer(32 * out_h2 * out_w2, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


def run_classification_benchmark(seeds=SEEDS):
    print(f"\n{'-'*20} BENCHMARK: DEEP CNN (OLIVETTI FACES) {'-'*20}")

    faces = fetch_olivetti_faces()
    X, y = faces.data, faces.target
    img_h, img_w = 64, 64
    num_classes = 40

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=0.2, random_state=42
    )

    X_train_cnn = X_train.reshape(-1, 1, img_h, img_w)
    X_test_cnn = X_test.reshape(-1, 1, img_h, img_w)
    y_train_onehot = np.eye(num_classes, dtype=np.float32)[y_train]

    epochs = 40
    batch_size = 32
    lr = 0.001

    # --- 1. PyTorch ---
    print("Evaluating PyTorch (ATen) across seeds...")
    best_pt_acc = -1.0
    best_pt_results = None

    for seed in seeds:
        torch.manual_seed(seed)
        pt_model = PyTorchDeepCNN(img_h, img_w, num_classes).to(device)
        pt_opt = optim.Adam(pt_model.parameters(), lr=lr)
        pt_loss_fn = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.tensor(X_train_cnn), torch.tensor(y_train))
        train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        t0 = time.perf_counter()
        pt_model.train()
        for epoch in range(epochs):
            for bx, by in train_loader:
                pt_opt.zero_grad()
                loss = pt_loss_fn(pt_model(bx), by)
                loss.backward()
                pt_opt.step()
        pt_time = time.perf_counter() - t0

        pt_model.eval()
        with torch.no_grad():
            pt_preds = torch.argmax(pt_model(torch.tensor(X_test_cnn)), dim=1).numpy()
        pt_acc = accuracy_score(y_test, pt_preds) * 100

        if pt_acc > best_pt_acc:
            best_pt_acc = pt_acc
            best_pt_results = (pt_acc, pt_time)
    pt_acc, pt_time = best_pt_results

    # --- 2. AffineFlow-NN JIT ---
    print("Evaluating AffineFlow-NN (JIT) across seeds...")
    best_nn_acc = -1.0
    best_nn_results = None

    for seed in seeds:
        afnn.set_seed(seed)
        nn_model = AffineFlowNNDeepCNN(img_h, img_w, num_classes)
        nn_opt = afnn.Adam(learning_rate=lr)
        nn_loss = afnn.SoftmaxCrossEntropyLoss()

        trainer = afnn.JITCompiler(nn_model, nn_opt, nn_loss)
        nn_loader = afnn.DataLoader(X_train_cnn, y_train_onehot, batch_size=batch_size, shuffle=True, drop_last=True)

        t0 = time.perf_counter()
        trainer.fit(nn_loader, epochs=epochs, verbose=False)
        nn_time = time.perf_counter() - t0

        nn_preds = np.argmax(np.array(nn_model.predict(X_test_cnn)), axis=1)
        nn_acc = accuracy_score(y_test, nn_preds) * 100

        if nn_acc > best_nn_acc:
            best_nn_acc = nn_acc
            best_nn_results = (nn_acc, nn_time)
    nn_acc, nn_time = best_nn_results

    # --- 3. AffineFlow-NN Eager (PyTorch-like Implicit Execution) ---
    print("Evaluating AffineFlow-NN (Eager) across seeds...")
    best_eag_acc = -1.0
    best_eag_results = None

    for seed in seeds:
        afnn.set_seed(seed)
        eag_model = AffineFlowNNDeepCNN(img_h, img_w, num_classes)
        eag_opt = afnn.Adam(learning_rate=lr)
        eag_opt.set_parameters(eag_model.parameters())
        eag_loss = afnn.SoftmaxCrossEntropyLoss()

        eag_loader = afnn.DataLoader(X_train_cnn, y_train_onehot, batch_size=batch_size, shuffle=True, drop_last=True)

        bx = afnn.Tensor(np.zeros((batch_size, 1, img_h, img_w), dtype=np.float32))
        by = afnn.Tensor(np.zeros((batch_size, num_classes), dtype=np.float32))

        t0 = time.perf_counter()
        eag_model.train(True)
        for epoch in range(epochs):
            eag_loader.reset()
            while eag_loader.has_next():
                eag_loader.next_batch(bx, by)

                eag_opt.zero_grad()
                preds = eag_model(bx)
                loss = eag_loss.forward(preds, by)
                eag_loss.backward()
                eag_opt.step()

        eag_time = time.perf_counter() - t0

        eag_preds = np.argmax(np.array(eag_model.predict(X_test_cnn)), axis=1)
        eag_acc = accuracy_score(y_test, eag_preds) * 100

        if eag_acc > best_eag_acc:
            best_eag_acc = eag_acc
            best_eag_results = (eag_acc, eag_time)
    eag_acc, eag_time = best_eag_results

    # Results
    print(f"\n--- Classification Results (Best over {len(seeds)} Seeds) ---")
    print(f"PyTorch           | Acc: {pt_acc:.2f}% | Time: {pt_time:.4f}s")
    print(f"AffineFlow-NN JIT | Acc: {nn_acc:.2f}% | Time: {nn_time:.4f}s")
    print(f"AffineFlow-NN Eag | Acc: {eag_acc:.2f}% | Time: {eag_time:.4f}s")


if __name__ == "__main__":
    print("Starting Comprehensive Multi-Seed AffineFlow Validation Suite...")
    run_knn_classification()
    run_regression_benchmark()
    run_classification_benchmark()
