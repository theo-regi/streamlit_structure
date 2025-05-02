import numpy as np
from scipy.optimize import minimize
from constants import INITIAL_NS, SOLVER_METHOD

def nelson_siegel(t:float, beta0:float, beta1:float, beta2:float, lam:float)-> float:
    """Nelson-Siegel model function."""
    return beta0 + beta1 * (1 - (np.exp(-t*lam) / (lam*t))) + beta2 * (((1 - np.exp(-lam*t)) / (lam*t)) - np.exp(-t / lam))

def objective_function(params:list, t:float, y:float) -> float:
    """Objective function to minimize."""
    beta0, beta1, beta2, lam = params
    y_hat = nelson_siegel(t, beta0, beta1, beta2, lam)
    return np.sum((y - y_hat) ** 2)

def optimize_nelson_siegel(maturities, rates) -> np.array:
    """Optimize Nelson-Siegel parameters."""
    result = minimize(objective_function, INITIAL_NS, args=(maturities, rates), method=SOLVER_METHOD)
    return result.x