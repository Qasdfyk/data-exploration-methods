import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml, make_classification
from data_loader import DatasetLoader
from preprocessing import Preprocessor
from experiments import ExperimentRunner
from visualization import plot_metric, plot_bar_chart, plot_multiple_lines

def get_datasets():
    datasets = {}
    loader = DatasetLoader()
    
    print("Pobieranie Breast Cancer Wisconsin...")
    bc = load_breast_cancer()
    df_bc = pd.DataFrame(bc.data, columns=bc.feature_names)
    datasets['Breast Cancer'] = loader.to_numpy(loader.select_numeric_columns(df_bc))
    
    print("Pobieranie Adult / Census Income...")
    try:
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
        
        # 1. Baseline i Wykres Słupkowy
        print("--- Uruchamianie testu baseline ---")
        baseline_results = runner.run_baseline_comparison(name, X)
        runner.save_results(baseline_results, f"results/baseline_{name.replace(' ', '_')}.csv")
        plot_bar_chart(baseline_results, x="method", y="rmse", 
                       output_path=f"results/baseline_bar_{name.replace(' ', '_')}.png", 
                       title=f"Porównanie Metod Imputacji (RMSE) - {name}")
        
        # 2. Jakość (Wpływ braków i parametr K na RMSE i MAE)
        print("--- Uruchamianie eksperymentu jakości ---")
        missing_rates = [0.05, 0.15, 0.30]
        k_values = [3, 5, 7, 11]
        
        X_qual = X if X.shape[0] <= 1000 else X[:1000]
            
        quality_results = runner.run_quality_experiment(name, X_qual, missing_rates, k_values)
        runner.save_results(quality_results, f"results/quality_{name.replace(' ', '_')}.csv")
        
        # Dla wybranej wartości k=5 rysujemy RMSE i MAE razem
        k5_results = quality_results[quality_results['k'] == 5]
        plot_multiple_lines(k5_results, x="missing_rate", y_cols=["rmse", "mae"], labels=["RMSE", "MAE"],
                            output_path=f"results/quality_rmse_mae_{name.replace(' ', '_')}.png",
                            title=f"RMSE vs MAE (K=5) - {name}")
        
        # Stary wykres wpływu K na RMSE
        plot_metric(quality_results, x="missing_rate", y="rmse", group="k", 
                    output_path=f"results/quality_rmse_k_{name.replace(' ', '_')}.png", 
                    title=f"Wpływ braków danych i parametru k na RMSE ({name})")
                    
        # 3. Wpływ metody agregacji (Uniform vs Distance)
        print("--- Uruchamianie eksperymentu wag (Uniform vs Distance) ---")
        weights_results = runner.run_weights_comparison(name, X_qual, missing_rate=0.15, k_values=k_values)
        runner.save_results(weights_results, f"results/weights_{name.replace(' ', '_')}.csv")
        plot_metric(weights_results, x="k", y="rmse", group="weight",
                    output_path=f"results/weights_comparison_{name.replace(' ', '_')}.png",
                    title=f"Uniform vs Distance Weighting ({name})")
                    
        # 4. Wpływ parametru K na skalowalność czasu
        print("--- Uruchamianie skalowalności parametru K ---")
        k_scale_results = runner.run_k_scalability_experiment(name, X, k_values=[3, 5, 9, 15, 21])
        runner.save_results(k_scale_results, f"results/scalability_k_{name.replace(' ', '_')}.csv")
        plot_metric(k_scale_results, x="k", y="time_seconds", group="dataset",
                    output_path=f"results/scalability_k_plot_{name.replace(' ', '_')}.png",
                    title=f"Czas działania vs parametr K ({name})")

        # 5. Skalowalność (Wiersze / Kolumny)
        print("--- Uruchamianie eksperymentu skalowalności objętościowej ---")
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
