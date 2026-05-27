import numpy as np

class Preprocessor:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit_minmax(self, X: np.ndarray) -> None:
        """Oblicza minima i maksima kolumn ignorując wartości NaN."""
        self.min_vals = np.nanmin(X, axis=0)
        self.max_vals = np.nanmax(X, axis=0)

    def transform_minmax(self, X: np.ndarray) -> np.ndarray:
        """Normalizuje wartości macierzy do przedziału [0, 1]."""
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Brak wyliczonych wartości min/max. Użyj fit_minmax przed transformacją.")
            
        diff = self.max_vals - self.min_vals
        # Zabezpieczenie przed dzieleniem przez zero dla stałych kolumn
        diff = np.where(diff == 0, 1.0, diff)
        
        return (X - self.min_vals) / diff

    def inverse_transform_minmax(self, X: np.ndarray) -> np.ndarray:
        """Przywraca znormalizowane dane do ich oryginalnej skali."""
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Brak wyliczonych wartości min/max. Użyj fit_minmax przed transformacją odwrotną.")
            
        diff = self.max_vals - self.min_vals
        diff = np.where(diff == 0, 1.0, diff)
        
        return X * diff + self.min_vals

    def build_missing_mask(self, X: np.ndarray) -> np.ndarray:
        """Tworzy maskę logiczną wkazującą obecność wartości brakujących (True dla NaN)."""
        return np.isnan(X)

    def inject_missing_values(self, X: np.ndarray, missing_rate: float, seed: int = None) -> tuple[np.ndarray, np.ndarray]:
        """Losowo wstrzykuje braki (MCAR) do macierzy na podstawie wskazanego prawdopodobieństwa."""
        if seed is not None:
            np.random.seed(seed)
            
        mask = np.random.rand(*X.shape) < missing_rate
        X_missing = X.copy()
        X_missing[mask] = np.nan
        
        return X_missing, mask
