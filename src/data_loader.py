import pandas as pd
import numpy as np

class DatasetLoader:
    def load_csv(self, path: str) -> pd.DataFrame:
        """Wczytuje plik z danymi źródłowymi."""
        return pd.read_csv(path)

    def select_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wybiera kolumny numeryczne z pominięciem kategorycznych i tekstowych."""
        return df.select_dtypes(include=[np.number])

    def to_numpy(self, df: pd.DataFrame) -> np.ndarray:
        """Konwertuje dane tabelaryczne na dwuwymiarową macierz numeryczną numpy."""
        return df.to_numpy(dtype=float)
