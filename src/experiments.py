import pandas as pd
import numpy as np
import os
from knn_imputer import KNNImputer
from metrics import rmse, mae, measure_time
from sklearn.impute import KNNImputer as SklearnKNNImputer

class ExperimentRunner:
    def __init__(self, data_loader, preprocessor):
        self.data_loader = data_loader
        self.preprocessor = preprocessor

    def run_quality_experiment(self, dataset_name: str, X_true: np.ndarray, missing_rates: list[float], k_values: list[int]) -> pd.DataFrame:
        """Testuje jakość imputacji dla różnych poziomów braków i wartości k."""
        results = []
        
        for rate in missing_rates:
            X_missing, mask = self.preprocessor.inject_missing_values(X_true, rate, seed=42)
            
            for k in k_values:
                imputer = KNNImputer(k=k, weights='distance')
                X_imputed, duration = measure_time(imputer.fit_transform, X_missing)
                
                true_vals = X_true[mask]
                pred_vals = X_imputed[mask]
                
                err_rmse = rmse(true_vals, pred_vals)
                err_mae = mae(true_vals, pred_vals)
                
                results.append({
                    "dataset": dataset_name,
                    "missing_rate": rate,
                    "k": k,
                    "rmse": err_rmse,
                    "mae": err_mae,
                    "time_seconds": duration
                })
                
        return pd.DataFrame(results)

    def run_scalability_experiment(self, dataset_name: str, X_true: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> pd.DataFrame:
        """Mierzy czas działania dla rosnącej liczby rekordów i cech."""
        results = []
        rate = 0.1 # 10% braków
        k = 5
        
        for n in row_sizes:
            for m in col_sizes:
                if n > X_true.shape[0] or m > X_true.shape[1]:
                    continue
                    
                X_sub = X_true[:n, :m]
                X_missing, mask = self.preprocessor.inject_missing_values(X_sub, rate, seed=42)
                
                imputer = KNNImputer(k=k, weights='distance')
                _, duration = measure_time(imputer.fit_transform, X_missing)
                
                results.append({
                    "dataset": dataset_name,
                    "n_rows": n,
                    "m_cols": m,
                    "time_seconds": duration
                })
                
        return pd.DataFrame(results)

    def run_baseline_comparison(self, dataset_name: str, X_true: np.ndarray) -> pd.DataFrame:
        """Porównuje wynik z imputacją średnią/medianą oraz KNNImputer ze scikit-learn."""
        results = []
        rate = 0.2
        X_missing, mask = self.preprocessor.inject_missing_values(X_true, rate, seed=42)
        true_vals = X_true[mask]
        
        # Srednia kolumny
        X_mean = X_missing.copy()
        for j in range(X_mean.shape[1]):
            col = X_mean[:, j]
            mean_val = np.nanmean(col)
            X_mean[np.isnan(col), j] = mean_val if not np.isnan(mean_val) else 0.0
            
        err_rmse = rmse(true_vals, X_mean[mask])
        results.append({"dataset": dataset_name, "method": "Mean Imputation", "rmse": err_rmse, "time_seconds": 0.0})
        
        # Mediana kolumny
        X_median = X_missing.copy()
        for j in range(X_median.shape[1]):
            col = X_median[:, j]
            median_val = np.nanmedian(col)
            X_median[np.isnan(col), j] = median_val if not np.isnan(median_val) else 0.0
            
        err_rmse = rmse(true_vals, X_median[mask])
        results.append({"dataset": dataset_name, "method": "Median Imputation", "rmse": err_rmse, "time_seconds": 0.0})
        
        # Autorski KNN
        imputer = KNNImputer(k=5, weights='distance')
        X_knn, duration = measure_time(imputer.fit_transform, X_missing)
        err_rmse = rmse(true_vals, X_knn[mask])
        results.append({"dataset": dataset_name, "method": "Custom KNN (k=5)", "rmse": err_rmse, "time_seconds": duration})
        
        # Sklearn KNN
        sk_imputer = SklearnKNNImputer(n_neighbors=5, weights='distance')
        X_sk, duration = measure_time(sk_imputer.fit_transform, X_missing)
        err_rmse = rmse(true_vals, X_sk[mask])
        results.append({"dataset": dataset_name, "method": "Sklearn KNN (k=5)", "rmse": err_rmse, "time_seconds": duration})
        
        return pd.DataFrame(results)

    def run_weights_comparison(self, dataset_name: str, X_true: np.ndarray, missing_rate: float, k_values: list[int]) -> pd.DataFrame:
        """Porównuje błąd dla metod agregacji (uniform vs distance)."""
        results = []
        X_missing, mask = self.preprocessor.inject_missing_values(X_true, missing_rate, seed=42)
        true_vals = X_true[mask]
        
        for k in k_values:
            for weight in ['uniform', 'distance']:
                imputer = KNNImputer(k=k, weights=weight)
                X_imputed, duration = measure_time(imputer.fit_transform, X_missing)
                err_rmse = rmse(true_vals, X_imputed[mask])
                results.append({
                    "dataset": dataset_name,
                    "k": k,
                    "weight": weight,
                    "rmse": err_rmse,
                    "time_seconds": duration
                })
        return pd.DataFrame(results)

    def run_k_scalability_experiment(self, dataset_name: str, X_true: np.ndarray, k_values: list[int]) -> pd.DataFrame:
        """Bada narzut czasowy związany z rosnącym parametrem K."""
        results = []
        X_sub = X_true[:min(500, X_true.shape[0])]
        X_missing, _ = self.preprocessor.inject_missing_values(X_sub, 0.1, seed=42)
        
        for k in k_values:
            imputer = KNNImputer(k=k, weights='distance')
            _, duration = measure_time(imputer.fit_transform, X_missing)
            results.append({
                "dataset": dataset_name,
                "k": k,
                "time_seconds": duration
            })
        return pd.DataFrame(results)

    def save_results(self, results: pd.DataFrame, path: str) -> None:
        """Zapisuje wyniki do pliku CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results.to_csv(path, index=False)
