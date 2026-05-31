import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Dodajemy folder 'src' do ścieżki Pythona, żeby poprawnie zaimportować moduły
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from data_loader import DatasetLoader
from preprocessing import Preprocessor
from knn_imputer import KNNImputer
from metrics import rmse, mae, measure_time

def main():
    print("Inicjalizacja dema...")
    loader = DatasetLoader()
    preprocessor = Preprocessor()
    
    print("\nPobieranie małego zbioru (Breast Cancer Wisconsin - ok. 569 wierszy, 30 cech)...")
    bc = load_breast_cancer()
    df_bc = pd.DataFrame(bc.data, columns=bc.feature_names)
    X_true = loader.to_numpy(loader.select_numeric_columns(df_bc))
    
    print(f"Rozmiar danych: {X_true.shape[0]} wierszy, {X_true.shape[1]} cech.")
    
    print("\nNormalizacja danych (Min-Max)...")
    preprocessor.fit_minmax(X_true)
    X_norm = preprocessor.transform_minmax(X_true)
    
    missing_rate = 0.15
    print(f"\nWstrzykiwanie sztucznych braków danych ({int(missing_rate*100)}% zniknie)...")
    X_missing, mask = preprocessor.inject_missing_values(X_norm, missing_rate, seed=42)
    
    print("\nUruchamianie algorytmu kNN Imputer (k=5, wagi=distance)...")
    imputer = KNNImputer(k=5, weights='distance')
    
    # Mierzymy czas i imputujemy
    X_imputed, duration = measure_time(imputer.fit_transform, X_missing)
    
    # Odwracamy normalizację żeby pokazać wartości w oryginalnej skali
    X_imputed_original_scale = preprocessor.inverse_transform_minmax(X_imputed)
    X_true_original_scale = preprocessor.inverse_transform_minmax(X_norm)
    
    true_vals = X_norm[mask]
    pred_vals = X_imputed[mask]
    
    err_rmse = rmse(true_vals, pred_vals)
    err_mae = mae(true_vals, pred_vals)
    
    print("\n" + "="*40)
    print("--- WYNIKI DEMO NA ŻYWO ---")
    print("="*40)
    print(f"Czas wykonania: {duration:.4f} sekund  <-- Prawie natychmiast!")
    print(f"Błąd RMSE:      {err_rmse:.4f}")
    print(f"Błąd MAE:       {err_mae:.4f}")
    
    print("\nPrzykładowe uzupełnione wartości (w oryginalnej skali):")
    missing_indices = np.argwhere(mask)
    for i in range(min(5, len(missing_indices))):
        row, col = missing_indices[i]
        val_true = X_true_original_scale[row, col]
        val_pred = X_imputed_original_scale[row, col]
        print(f"Wiersz {row}, Cecha '{bc.feature_names[col]}' -> Prawdziwa: {val_true:.4f}, Zgadnięta: {val_pred:.4f}")
        
    print("\nGotowe!")

if __name__ == "__main__":
    main()
