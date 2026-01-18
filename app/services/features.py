import numpy as np
import pandas as pd
from typing import Dict, Tuple

class FeatureEngineering:
    """
    Feature engineering service that replicates the logic from mining2.ipynb
    Generates features: enq1, enq2, enq3, enq4, enq5, enq6, enq7, SD
    """
    
    def __init__(self):
        self.feature_params = {}  # Will store optimized parameters
    
    def compute_enq1_base(self, W: np.ndarray, R: np.ndarray, b: float) -> np.ndarray:
        """Base function for enq1: (R / sqrt(W))^(-b)"""
        return (R / np.sqrt(W)) ** (-b)
    
    def optimize_enq1(self, W: np.ndarray, R: np.ndarray, actual_ppv: np.ndarray) -> Tuple[float, float]:
        """Optimize b and k for enq1"""
        b_values = np.arange(-2.99, 3.0, 0.01)
        b_values = b_values[~np.isclose(b_values, 0)]
        results = []
        
        for b in b_values:
            calc = self.compute_enq1_base(W, R, b)
            corr = np.corrcoef(calc, actual_ppv)[0, 1]
            results.append((b, corr))
        
        best_b, best_corr = max(results, key=lambda x: abs(x[1]))
        
        # Optimize k using least squares
        F = self.compute_enq1_base(W, R, best_b)
        best_k = np.sum(F * actual_ppv) / np.sum(F ** 2)
        
        return best_b, best_k
    
    def compute_enq1(self, W: np.ndarray, R: np.ndarray, k: float, b: float) -> np.ndarray:
        """enq1: k * (R / sqrt(W))^(-b)"""
        return k * (R / np.sqrt(W)) ** (-b)
    
    def compute_enq2_base(self, W: np.ndarray, R: np.ndarray, b: float) -> np.ndarray:
        """Base function for enq2: (W / (R^(2/3)))^(b/2)"""
        return (W / (R ** (2/3))) ** (b / 2)
    
    def optimize_enq2(self, W: np.ndarray, R: np.ndarray, actual_ppv: np.ndarray) -> Tuple[float, float]:
        """Optimize b and k for enq2"""
        b_values = np.arange(-2.99, 3.0, 0.01)
        b_values = b_values[~np.isclose(b_values, 0)]
        results = []
        
        for b in b_values:
            calc = self.compute_enq2_base(W, R, b)
            corr = np.corrcoef(calc, actual_ppv)[0, 1]
            results.append((b, corr))
        
        best_b, best_corr = max(results, key=lambda x: abs(x[1]))
        
        # Optimize k
        F = self.compute_enq2_base(W, R, best_b)
        best_k = np.sum(F * actual_ppv) / np.sum(F ** 2)
        
        return best_b, best_k
    
    def compute_enq2(self, W: np.ndarray, R: np.ndarray, k: float, b: float) -> np.ndarray:
        """enq2: k * (W / (R^(2/3)))^(b/2)"""
        return k * (W / (R ** (2/3))) ** (b / 2)
    
    def compute_enq3_base(self, W: np.ndarray, R: np.ndarray, b: float) -> np.ndarray:
        """Base function for enq3: (R / cbrt(W))^(-b)"""
        return (R / np.cbrt(W)) ** (-b)
    
    def optimize_enq3(self, W: np.ndarray, R: np.ndarray, actual_ppv: np.ndarray) -> Tuple[float, float]:
        """Optimize b and k for enq3"""
        b_values = np.arange(-2.99, 3.0, 0.01)
        b_values = b_values[np.abs(b_values) > 1e-9]  # Removes 0
        results = []
        
        for b in b_values:
            calc = self.compute_enq3_base(W, R, b)
            corr = np.corrcoef(calc, actual_ppv)[0, 1]
            results.append((b, corr))
        
        best_b, best_corr = max(results, key=lambda x: abs(x[1]))
        
        # Optimize k
        F = self.compute_enq3_base(W, R, best_b)
        best_k = np.sum(F * actual_ppv) / np.sum(F ** 2)
        
        return best_b, best_k
    
    def compute_enq3(self, W: np.ndarray, R: np.ndarray, k: float, b: float) -> np.ndarray:
        """enq3: k * (R / cbrt(W))^(-b)"""
        return k * (R / np.cbrt(W)) ** (-b)
    
    def compute_enq4_base(self, W: np.ndarray, R: np.ndarray, b: float) -> np.ndarray:
        """Base function for enq4: ((R^(2/3)) / W)^(-b)"""
        return ((R ** (2/3)) / W) ** (-b)
    
    def optimize_enq4(self, W: np.ndarray, R: np.ndarray, actual_ppv: np.ndarray) -> Tuple[float, float]:
        """Optimize b and k for enq4"""
        b_values = np.arange(-2.99, 3.0, 0.01)
        b_values = b_values[~np.isclose(b_values, 0)]
        results = []
        
        for b in b_values:
            calc = self.compute_enq4_base(W, R, b)
            corr = np.corrcoef(calc, actual_ppv)[0, 1]
            results.append((b, corr))
        
        best_b, best_corr = max(results, key=lambda x: abs(x[1]))
        
        # Optimize k
        F = self.compute_enq4_base(W, R, best_b)
        best_k = np.sum(F * actual_ppv) / np.sum(F ** 2)
        
        return best_b, best_k
    
    def compute_enq4(self, W: np.ndarray, R: np.ndarray, k: float, b: float) -> np.ndarray:
        """enq4: k * ((R^(2/3)) / W)^(-b)"""
        return k * ((R ** (2/3)) / W) ** (-b)
    
    def compute_ppv_ado(self, W: np.ndarray, R: np.ndarray, b: float, alpha: float, k: float) -> np.ndarray:
        """Compute PPV using ADO-optimized parameters: k * (R / sqrt(W))^(-b) * exp(-alpha * R)"""
        return k * (R / np.sqrt(W)) ** (-b) * np.exp(-alpha * R)
    
    def fitness_function_ado(self, params: Tuple[float, float], W: np.ndarray, R: np.ndarray, 
                            y_true: np.ndarray, k: float) -> float:
        """Fitness function for ADO optimization"""
        b, alpha = params
        y_pred = self.compute_ppv_ado(W, R, b, alpha, k)
        
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return 1e9
        
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred)
    
    def ado_optimize_ppv(self, W: np.ndarray, R: np.ndarray, y: np.ndarray, 
                        bounds: Dict[str, Tuple[float, float]],
                        k: float = 1.0,
                        M: int = 6, iterations: int = 30, K: int = 40,
                        s0: float = 0.3, T0: float = 1.0, c: int = 1,
                        p: float = 1.0, alpha_d: float = 1.0, beta: float = 1.0) -> Tuple[float, float]:
        """
        ADO (Adaptive Differential Optimization) algorithm for parameter optimization
        """
        import random
        
        def denormalize(norm_vec, bounds):
            b_range = bounds["b"]
            alpha_range = bounds["alpha"]
            b = b_range[0] + norm_vec[0] * (b_range[1] - b_range[0])
            alpha = alpha_range[0] + norm_vec[1] * (alpha_range[1] - alpha_range[0])
            return b, alpha
        
        # Initialize epicenters
        epicenters = [np.array([random.random(), random.random()]) for _ in range(M)]
        losses = []
        
        for ep in epicenters:
            params = denormalize(ep, bounds)
            loss = self.fitness_function_ado(params, W, R, y, k)
            losses.append(loss)
        
        losses = np.array(losses)
        best_idx = np.argmin(losses)
        best_ep = epicenters[best_idx].copy()
        best_loss = losses[best_idx]
        
        for t in range(iterations):
            Nt = int(np.ceil(K / ((c + t) ** p)))
            step = s0 / ((c + t) ** alpha_d)
            Tt = T0 / ((c + t) ** beta)
            
            for i in range(M):
                for _ in range(Nt):
                    delta = np.random.normal(0, 1, 2)
                    candidate = np.clip(epicenters[i] + step * delta, 0, 1)
                    
                    cand_params = denormalize(candidate, bounds)
                    cand_loss = self.fitness_function_ado(cand_params, W, R, y, k)
                    
                    # Metropolis acceptance
                    if cand_loss < losses[i]:
                        accept = True
                    else:
                        prob = np.exp(-(cand_loss - losses[i]) / max(Tt, 1e-8))
                        accept = random.random() < prob
                    
                    if accept:
                        epicenters[i] = candidate
                        losses[i] = cand_loss
                        
                        if cand_loss < best_loss:
                            best_loss = cand_loss
                            best_ep = candidate.copy()
        
        best_params = denormalize(best_ep, bounds)
        return best_params
    
    def optimize_enq5(self, W: np.ndarray, R: np.ndarray, actual_ppv: np.ndarray, k: float) -> Tuple[float, float, float]:
        """Optimize b, alpha, and k for enq5 using grid search"""
        def compute_enq5_base(W, R, b, alpha):
            return (R / np.sqrt(W)) ** (-b) * np.exp(-alpha * R)
        
        # Search ranges
        b_values = np.arange(-2.99, 3.0, 0.05)
        b_values = b_values[~np.isclose(b_values, 0)]
        alpha_values = np.arange(0.001, 1.0, 0.01)
        
        results = []
        
        for b in b_values:
            for alpha in alpha_values:
                calc = compute_enq5_base(W, R, b, alpha)
                corr = np.corrcoef(calc, actual_ppv)[0, 1]
                results.append((b, alpha, corr))
        
        # Best (b, alpha)
        best_b, best_alpha, best_corr = max(results, key=lambda x: abs(x[2]))
        
        # Optimize k
        F = (R / np.sqrt(W)) ** (-best_b) * np.exp(-best_alpha * R)
        best_k = np.sum(F * actual_ppv) / np.sum(F ** 2)
        
        return best_b, best_alpha, best_k
    
    def compute_enq5(self, W: np.ndarray, R: np.ndarray, k: float, b: float, alpha: float) -> np.ndarray:
        """enq5: k * (R / sqrt(W))^(-b) * exp(-alpha * R)"""
        return k * (R / np.sqrt(W)) ** (-b) * np.exp(-alpha * R)
    
    def optimize_enq6(self, W: np.ndarray, R: np.ndarray, actual_ppv: np.ndarray, k: float) -> Tuple[float, float, float]:
        """Optimize b, alpha, and k for enq6 using grid search"""
        def compute_enq6_base(W, R, b, alpha):
            return (R / np.cbrt(W)) ** (-b) * np.exp(-alpha * R)
        
        # Search ranges
        b_values = np.arange(-2.99, 3.0, 0.05)
        b_values = b_values[~np.isclose(b_values, 0)]
        alpha_values = np.arange(0.001, 1.0, 0.01)
        
        results = []
        
        for b in b_values:
            for alpha in alpha_values:
                calc = compute_enq6_base(W, R, b, alpha)
                corr = np.corrcoef(calc, actual_ppv)[0, 1]
                results.append((b, alpha, corr))
        
        # Best (b, alpha)
        best_b, best_alpha, best_corr = max(results, key=lambda x: abs(x[2]))
        
        # Optimize k
        F = (R / np.cbrt(W)) ** (-best_b) * np.exp(-best_alpha * R)
        best_k = np.sum(F * actual_ppv) / np.sum(F ** 2)
        
        return best_b, best_alpha, best_k
    
    def compute_enq6(self, W: np.ndarray, R: np.ndarray, k: float, b: float, alpha: float) -> np.ndarray:
        """enq6: k * (R / cbrt(W))^(-b) * exp(-alpha * R)"""
        return k * (R / np.cbrt(W)) ** (-b) * np.exp(-alpha * R)
    
    def optimize_enq7(self, W: np.ndarray, R: np.ndarray, actual_ppv: np.ndarray, k: float) -> Tuple[float, float, float]:
        """Optimize b, alpha, and k for enq7 using grid search"""
        def compute_enq7_base(W, R, b, alpha):
            return (R / np.sqrt(W)) ** b * np.exp(alpha * (R / W))
        
        # Search ranges
        b_values = np.arange(-2.99, 3.0, 0.05)
        b_values = b_values[~np.isclose(b_values, 0)]
        alpha_values = np.arange(0.001, 1.0, 0.01)
        
        results = []
        
        for b in b_values:
            for alpha in alpha_values:
                calc = compute_enq7_base(W, R, b, alpha)
                corr = np.corrcoef(calc, actual_ppv)[0, 1]
                results.append((b, alpha, corr))
        
        # Best (b, alpha)
        best_b, best_alpha, best_corr = max(results, key=lambda x: abs(x[2]))
        
        # Optimize k
        F = (R / np.sqrt(W)) ** best_b * np.exp(best_alpha * (R / W))
        best_k = np.sum(F * actual_ppv) / np.sum(F ** 2)
        
        return best_b, best_alpha, best_k
    
    def compute_enq7(self, W: np.ndarray, R: np.ndarray, k: float, b: float, alpha: float) -> np.ndarray:
        """enq7: k * (R / sqrt(W))^b * exp(alpha * (R / W))"""
        return k * (R / np.sqrt(W)) ** b * np.exp(alpha * (R / W))
    
    
    def compute_SD(self, W: np.ndarray, R: np.ndarray) -> np.ndarray:
        """SD: R / sqrt(W)"""
        return R / np.sqrt(W)
    
    def generate_all_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate all features for a dataframe with R, W, PPV columns.
        Optimizes parameters using actual_ppv, then computes features.
        Returns dict with feature arrays and optimized parameters.
        """
        W = df["W"].values
        R = df["R"].values
        actual_ppv = df["PPV"].values
        
        features = {}
        params = {}
        
        # Remove zero values (as in notebook)
        mask = (df != 0).all(axis=1)
        W_clean = W[mask]
        R_clean = R[mask]
        actual_ppv_clean = actual_ppv[mask]
        
        # SD (computed first to match notebook order: W, R, SD, enq1...)
        features["SD"] = self.compute_SD(W, R)
        
        # Optimize and compute each feature
        # enq1
        b1, k1 = self.optimize_enq1(W_clean, R_clean, actual_ppv_clean)
        features["enq1"] = self.compute_enq1(W, R, k1, b1)
        params["enq1"] = {"b": float(b1), "k": float(k1)}
        
        # enq2
        b2, k2 = self.optimize_enq2(W_clean, R_clean, actual_ppv_clean)
        features["enq2"] = self.compute_enq2(W, R, k2, b2)
        params["enq2"] = {"b": float(b2), "k": float(k2)}
        
        # enq3
        b3, k3 = self.optimize_enq3(W_clean, R_clean, actual_ppv_clean)
        features["enq3"] = self.compute_enq3(W, R, k3, b3)
        params["enq3"] = {"b": float(b3), "k": float(k3)}
        
        # enq4
        b4, k4 = self.optimize_enq4(W_clean, R_clean, actual_ppv_clean)
        features["enq4"] = self.compute_enq4(W, R, k4, b4)
        params["enq4"] = {"b": float(b4), "k": float(k4)}
        
        # enq5
        b5, alpha5, k5 = self.optimize_enq5(W_clean, R_clean, actual_ppv_clean, k=1.0)
        features["enq5"] = self.compute_enq5(W, R, k5, b5, alpha5)
        params["enq5"] = {"b": float(b5), "alpha": float(alpha5), "k": float(k5)}
        
        # enq6
        b6, alpha6, k6 = self.optimize_enq6(W_clean, R_clean, actual_ppv_clean, k=1.0)
        features["enq6"] = self.compute_enq6(W, R, k6, b6, alpha6)
        params["enq6"] = {"b": float(b6), "alpha": float(alpha6), "k": float(k6)}
        
        # enq7 (formerly enqX)
        b7, alpha7, k7 = self.optimize_enq7(W_clean, R_clean, actual_ppv_clean, k=1.0)
        features["enq7"] = self.compute_enq7(W, R, k7, b7, alpha7)
        params["enq7"] = {"b": float(b7), "alpha": float(alpha7), "k": float(k7)}
        
        return features, params
    
    def compute_features_from_params(self, W: float, R: float, params: Dict) -> Dict[str, float]:
        """
        Compute all features for a single R, W pair using saved parameters.
        """
        features = {}
        
        # W and R are features themselves
        features["W"] = float(W)
        features["R"] = float(R)
        
        # enq1
        p1 = params.get("enq1", {})
        features["enq1"] = float(self.compute_enq1(np.array([W]), np.array([R]), 
                                                   p1.get("k", 1.0), p1.get("b", 1.0))[0])
        
        # enq2
        p2 = params.get("enq2", {})
        features["enq2"] = float(self.compute_enq2(np.array([W]), np.array([R]), 
                                                   p2.get("k", 1.0), p2.get("b", 1.0))[0])
        
        # enq3
        p3 = params.get("enq3", {})
        features["enq3"] = float(self.compute_enq3(np.array([W]), np.array([R]), 
                                                   p3.get("k", 1.0), p3.get("b", 1.0))[0])
        
        # enq4
        p4 = params.get("enq4", {})
        features["enq4"] = float(self.compute_enq4(np.array([W]), np.array([R]), 
                                                   p4.get("k", 1.0), p4.get("b", 1.0))[0])
        
        # enq5
        p5 = params.get("enq5", {})
        features["enq5"] = float(self.compute_enq5(np.array([W]), np.array([R]), 
                                                   p5.get("k", 1.0), p5.get("b", 1.0), 
                                                   p5.get("alpha", 0.1))[0])
        
        # enq6
        p6 = params.get("enq6", {})
        features["enq6"] = float(self.compute_enq6(np.array([W]), np.array([R]), 
                                                   p6.get("k", 1.0), p6.get("b", 1.0), 
                                                   p6.get("alpha", 0.1))[0])
        
        # enq7 (formerly enqX)
        p7 = params.get("enq7", {})
        features["enq7"] = float(self.compute_enq7(np.array([W]), np.array([R]), 
                                                   p7.get("k", 1.0), p7.get("b", 1.0), 
                                                   p7.get("alpha", 0.1))[0])
        
        # SD
        features["SD"] = float(self.compute_SD(np.array([W]), np.array([R]))[0])
        
        return features

