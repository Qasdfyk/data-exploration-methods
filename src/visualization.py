import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metric(results: pd.DataFrame, x: str, y: str, group: str, output_path: str, title: str = "") -> None:
    """Generuje wykres liniowy i zapisuje go do pliku."""
    plt.figure(figsize=(10, 6))
    
    if group in results.columns:
        for key, grp in results.groupby(group):
            plt.plot(grp[x], grp[y], marker='o', label=f"{group}={key}")
    else:
        plt.plot(results[x], results[y], marker='o')
        
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
