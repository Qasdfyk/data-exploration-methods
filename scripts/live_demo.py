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
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    err_r2 = r2_score(true_vals, pred_vals)
    
    # Obliczanie Baseline (Średnia) dla porównania
    X_mean = X_missing.copy()
    for j in range(X_mean.shape[1]):
        col = X_mean[:, j]
        mean_val = np.nanmean(col)
        X_mean[np.isnan(col), j] = mean_val if not np.isnan(mean_val) else 0.0
    pred_vals_mean = X_mean[mask]
    err_rmse_mean = rmse(true_vals, pred_vals_mean)
    err_mae_mean = mae(true_vals, pred_vals_mean)
    
    print("\n" + "="*40)
    print("--- WYNIKI DEMO NA ŻYWO ---")
    print("="*40)
    print(f"Czas wykonania algorytmu kNN: {duration:.4f} sekund")
    
    print("\nPorównanie błędu RMSE (mniejszy błąd = wyższa dokładność):")
    print(f" -> Nasz algorytm kNN:           {err_rmse:.4f} (LEPIEJ)")
    print(f" -> Zwykłe wstawienie średniej:  {err_rmse_mean:.4f} (GORZEJ)")
    
    print("\nPorównanie błędu MAE (mniejszy błąd = wyższa dokładność):")
    print(f" -> Nasz algorytm kNN:           {err_mae:.4f} (LEPIEJ)")
    print(f" -> Zwykłe wstawienie średniej:  {err_mae_mean:.4f} (GORZEJ)")
    
    print("\nWspółczynnik R2 dla kNN (od 0.0 do 1.0, 1.0 = perfekcja):")
    print(f" -> Wynik dopasowania:           {err_r2:.4f}")
    
    print("\nPrzykładowe uzupełnione wartości (w oryginalnej skali):")
    missing_indices = np.argwhere(mask)
    for i in range(min(5, len(missing_indices))):
        row, col = missing_indices[i]
        val_true = X_true_original_scale[row, col]
        val_pred = X_imputed_original_scale[row, col]
        print(f"Wiersz {row}, Cecha '{bc.feature_names[col]}' -> Prawdziwa: {val_true:.4f}, Zgadnięta: {val_pred:.4f}")
        
    print("\n" + "="*40)
    print("--- WYNIKI KLASYFIKACJI ---")
    print("="*40)
    print("Sprawdzamy, jak imputacja wpływa na jakość klasyfikatora (Random Forest)...")
    
    y = bc.target
    X_train_clean, X_test_clean, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    X_train_mean, X_test_mean, _, _ = train_test_split(X_mean, y, test_size=0.3, random_state=42)
    X_train_knn, X_test_knn, _, _ = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    
    clf.fit(X_train_clean, y_train)
    acc_clean = accuracy_score(y_test, clf.predict(X_test_clean))
    
    clf.fit(X_train_knn, y_train)
    acc_knn = accuracy_score(y_test, clf.predict(X_test_knn))
    
    clf.fit(X_train_mean, y_train)
    acc_mean = accuracy_score(y_test, clf.predict(X_test_mean))

    print("\nDokładność (Accuracy) klasyfikacji:")
    print(f" -> 1. Czyste, oryginalne dane:    {acc_clean:.4f}")
    print(f" -> 2. Imputacja naszym kNN:       {acc_knn:.4f}")
    print(f" -> 3. Zwykłe wstawienie średniej: {acc_mean:.4f}")
    
    if acc_knn > acc_mean:
        print("\nWNIOSEK: Imputacja kNN pozwala zachować lepszą jakość danych dla modelu klasyfikacyjnego niż imputacja średnią!")
    else:
        print("\nWNIOSEK: W tym przypadku metoda średniej poradziła sobie nie gorzej niż kNN, ale często kNN wygrywa na trudniejszych zbiorach.")

    print("\nGotowe!")

if __name__ == "__main__":
    main()
