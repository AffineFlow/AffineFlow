import numpy as np

class Pipeline:
    """
    A Scikit-Learn style pipeline that sequentially applies a list of data transformers 
    and concludes with a final predictor. 
    
    Compatible natively with AffineFlow-ML transformers and AffineFlow-NN models 
    wrapped in an NNEstimator.
    """
    def __init__(self, steps):
        """
        Args:
            steps (list of tuples): List of (name, estimator) tuples.
        """
        self.steps = steps
        
    def fit(self, X, y=None):
        X_curr = X
        # Pass through all ML transformers
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                X_curr = step.fit_transform(X_curr)
            else:
                step.fit(X_curr)
                X_curr = step.transform(X_curr)
                
        # Fit the final ML/NN Estimator
        final_name, final_step = self.steps[-1]
        final_step.fit(X_curr, y)
        return self
        
    def predict(self, X):
        X_curr = X
        # Transform data through the pipeline
        for name, step in self.steps[:-1]:
            X_curr = step.transform(X_curr)
            
        # Predict using the final estimator
        final_name, final_step = self.steps[-1]
        return final_step.predict(X_curr)