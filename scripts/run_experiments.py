import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml, make_classification
from data_loader import DatasetLoader
from preprocessing import Preprocessor
from experiments import ExperimentRunner
from visualization import plot_metric

def get_datasets():
    datasets = {}
    loader = DatasetLoader()
    
    print("Pobieranie Breast Cancer Wisconsin...")
    bc = load_breast_cancer()
    df_bc = pd.DataFrame(bc.data, columns=bc.feature_names)
    datasets['Breast Cancer'] = loader.to_numpy(loader.select_numeric_columns(df_bc))
    
    print("Pobieranie Adult / Census Income...")
    try:
        # data_home="data" ensures it downloads to local data dir
        adult = fetch_openml(name='adult', version=2, as_frame=True, data_home="data")
        df_adult = adult.frame.dropna()
        if len(df_adult) > 10000:
            df_adult = df_adult.sample(10000, random_state=42)
        datasets['Adult'] = loader.to_numpy(loader.select_numeric_columns(df_adult))
    except Exception as e:
        print(f"Błąd pobierania Adult: {e}. Używam mock-data.")
        X, _ = make_classification(n_samples=10000, n_features=6, random_state=42)
        datasets['Adult'] = X
        
    print("Pobieranie Gene Expression Cancer RNA-Seq...")
    try:
        rna = fetch_openml(name='leukemia', version=1, as_frame=True, data_home="data")
        df_rna = rna.frame.dropna()
        datasets['RNA-Seq'] = loader.to_numpy(loader.select_numeric_columns(df_rna))
    except Exception as e:
        print(f"Błąd pobierania RNA-Seq: {e}. Używam mock-data.")
        X, _ = make_classification(n_samples=100, n_features=5000, random_state=42)
        datasets['RNA-Seq'] = X
        
    return datasets

def main():
    print("Inicjalizacja modułów...")
    loader = DatasetLoader()
    preprocessor = Preprocessor()
    runner = ExperimentRunner(loader, preprocessor)
    
    datasets = get_datasets()
    
    for name, X in datasets.items():
        print(f"\n{'='*40}")
        print(f"Rozpoczynanie eksperymentów dla zbioru: {name}")
        print(f"Rozmiar danych: {X.shape}")
        
        preprocessor.fit_minmax(X)
        X = preprocessor.transform_minmax(X)
        
        # 1. Baseline
        print("--- Uruchamianie testu baseline ---")
        baseline_results = runner.run_baseline_comparison(name, X)
        runner.save_results(baseline_results, f"results/baseline_{name.replace(' ', '_')}.csv")
        
        # 2. Jakość
        print("--- Uruchamianie eksperymentu jakości ---")
        missing_rates = [0.05, 0.15, 0.30]
        k_values = [3, 5, 7, 11]
        
        X_qual = X
        if X_qual.shape[0] > 1000:
            X_qual = X_qual[:1000]
            
        quality_results = runner.run_quality_experiment(name, X_qual, missing_rates, k_values)
        runner.save_results(quality_results, f"results/quality_{name.replace(' ', '_')}.csv")
        plot_metric(quality_results, x="missing_rate", y="rmse", group="k", 
                    output_path=f"results/quality_rmse_{name.replace(' ', '_')}.png", 
                    title=f"RMSE vs Missing Rate ({name})")
                    
        # 3. Skalowalność
        print("--- Uruchamianie eksperymentu skalowalności ---")
        if name == 'RNA-Seq':
            row_sizes = [min(50, X.shape[0])]
            col_sizes = [100, 500, 1000, 2000, min(5000, X.shape[1])]
        elif name == 'Adult':
            row_sizes = [500, 1000, 2000, 5000, X.shape[0]]
            col_sizes = [X.shape[1]]
        else:
            row_sizes = [100, 200, 300, 400, X.shape[0]]
            col_sizes = [X.shape[1]]
            
        scale_results = runner.run_scalability_experiment(name, X, row_sizes, col_sizes)
        runner.save_results(scale_results, f"results/scalability_{name.replace(' ', '_')}.csv")
        
        if name == 'RNA-Seq':
            plot_metric(scale_results, x="m_cols", y="time_seconds", group="n_rows", 
                        output_path=f"results/scalability_cols_{name.replace(' ', '_')}.png", 
                        title=f"Czas a liczba cech ({name})")
        else:
            plot_metric(scale_results, x="n_rows", y="time_seconds", group="m_cols", 
                        output_path=f"results/scalability_rows_{name.replace(' ', '_')}.png", 
                        title=f"Czas a liczba wierszy ({name})")
                
    print("\nZakończono. Wyniki zostały zapisane w katalogu results/.")

if __name__ == "__main__":
    main()
