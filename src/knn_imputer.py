import numpy as np

class KNNImputer:
    def __init__(self, k: int = 5, weights: str = 'uniform', epsilon: float = 1e-8):
        self.k = k
        self.weights = weights
        self.epsilon = epsilon
        self.X_ = None
        self.missing_mask_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Główna metoda uruchamiająca cały algorytm imputacji."""
        self.X_ = X.copy()
        self.missing_mask_ = np.isnan(self.X_)
        missing_cells = self._find_missing_cells(self.missing_mask_)

        for row_idx, col_idx in missing_cells:
            neighbors = self._find_neighbors(row_idx, col_idx)
            if not neighbors:
                # W przypadku braku poprawnych sąsiadów uzupełniamy średnią z kolumny
                val = self._fallback_column_value(col_idx)
            else:
                val = self._aggregate(neighbors)
            self.X_[row_idx, col_idx] = val

        return self.X_

    def _find_missing_cells(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Zwraca listę krotek zawierających indeksy (wiersz, kolumna) brakujących komórek."""
        row_indices, col_indices = np.where(mask)
        return list(zip(row_indices, col_indices))

    def _distance(self, row_a: np.ndarray, row_b: np.ndarray, target_col: int) -> float:
        """Oblicza zmodyfikowaną odległość euklidesową pomiędzy dwoma rekordami.
        W obliczeniach uwzględniane są tylko wspólne znane cechy, z pominięciem aktualnie imputowanej."""
        
        valid_a = ~np.isnan(row_a)
        valid_b = ~np.isnan(row_b)
        
        common_features = valid_a & valid_b
        common_features[target_col] = False  # Pomijamy cechę docelową
        
        if not np.any(common_features):
            return np.inf  # Brak wspólnych cech do policzenia odległości
            
        diff = row_a[common_features] - row_b[common_features]
        # Wzór: distance(a, b) = sqrt(sum((a_t - b_t)^2) / liczba_wspólnych_cech)
        return np.sqrt(np.sum(diff ** 2) / np.sum(common_features))

    def _find_neighbors(self, row_idx: int, col_idx: int) -> list[tuple[float, float]]:
        """Wyszukuje k najbliższych rekordów posiadających znaną wartość w imputowanej kolumnie."""
        target_row = self.X_[row_idx]
        distances = []
        
        for i in range(self.X_.shape[0]):
            if i == row_idx:
                continue  # Pomijamy ten sam rekord
                
            # Kandydat na sąsiada musi posiadać znaną wartość dla kolumny docelowej
            if np.isnan(self.X_[i, col_idx]):
                continue
                
            dist = self._distance(target_row, self.X_[i], col_idx)
            if dist != np.inf:
                # Zapisujemy parę: odległość i wartość cechy u sąsiada
                distances.append((dist, self.X_[i, col_idx]))
                
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    def _aggregate(self, neighbors: list[tuple[float, float]]) -> float:
        """Wylicza wartość zastępczą dla braku danych na podstawie znalezionych sąsiadów."""
        if self.weights == 'uniform':
            return float(np.mean([val for _, val in neighbors]))
        elif self.weights == 'distance':
            weights = []
            values = []
            for dist, val in neighbors:
                w = 1.0 / (dist + self.epsilon)
                weights.append(w)
                values.append(val)
            return float(np.sum(np.array(weights) * np.array(values)) / np.sum(weights))
        else:
            raise ValueError(f"Nieznany parametr wag: {self.weights}. Oczekiwano 'uniform' lub 'distance'.")

    def _fallback_column_value(self, col_idx: int) -> float:
        """Zwraca średnią z kolumny, gdy odnalezienie sąsiadów zawiedzie."""
        col_values = self.X_[:, col_idx]
        valid_values = col_values[~np.isnan(col_values)]
        if len(valid_values) == 0:
            return 0.0 # Jeśli cała kolumna jest NaN
        return float(np.mean(valid_values))
