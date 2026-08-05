import numpy as np
import affineflow_nn as nn

class NNEstimator:
    """
    A Scikit-Learn compatible adapter for AffineFlow-NN models.
    
    Wraps an nn.Module, Optimizer, and Loss function into a standard .fit() 
    and .predict() interface for seamless use inside an af.Pipeline.
    """
    def __init__(self, model, optimizer, loss_fn, regularizer=None, 
                 epochs=10, batch_size=32, shuffle=True, verbose=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.regularizer = regularizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        
        self.trainer = nn.JITCompiler(
            self.model, 
            self.optimizer, 
            self.loss_fn, 
            regularizer=self.regularizer
        )
        
    def fit(self, X, y):
        X_np = np.ascontiguousarray(X, dtype=np.float32)
        y_np = np.ascontiguousarray(y, dtype=np.float32)
        
        dataloader = nn.DataLoader(
            nn.Tensor(X_np), 
            nn.Tensor(y_np), 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
        
        self.trainer.fit(dataloader, epochs=self.epochs, verbose=self.verbose)
        return self

    def predict(self, X):
        X_np = np.ascontiguousarray(X, dtype=np.float32)
        X_tensor = nn.Tensor(X_np)
        
        # If it's a full Module, use the highly optimized native C++ inference
        if hasattr(self.model, "predict"):
            out_tensor = self.model.predict(X_tensor)
            return np.array(out_tensor, copy=False)
            
        # Fallback: Manually manage the evaluation graph for raw Layers
        self.model.eval()
        eval_tape = nn.Tape(record_ops=False)
        with eval_tape:
            # Push the tensor to the arena and execute the forward pass
            t_in = eval_tape.push_tensor(X_tensor, requires_grad=False)
            out_tensor = self.model.forward(t_in)
            
            # We must force a copy because the memory lives in the eval_tape arena,
            # which is immediately destroyed when the 'with' block exits.
            predictions = np.array(out_tensor, copy=True)
            
        self.model.train(True)
        return predictions