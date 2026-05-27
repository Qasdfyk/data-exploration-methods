import numpy as np
import time

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Błąd średniokwadratowy pierwiastkowy (Root Mean Squared Error)."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Średni błąd bezwzględny (Mean Absolute Error)."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def measure_time(function, *args, **kwargs) -> tuple[object, float]:
    """Mierzy czas wykonania wybranej funkcji i zwraca (wynik, czas_w_sekundach)."""
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    end_time = time.perf_counter()
    duration = end_time - start_time
    return result, duration
