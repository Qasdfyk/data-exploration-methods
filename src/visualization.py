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

def plot_bar_chart(results: pd.DataFrame, x: str, y: str, output_path: str, title: str = "") -> None:
    """Generuje i zapisuje wykres słupkowy, z uwzględnieniem etykiet na słupkach."""
    plt.figure(figsize=(10, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
    bars = plt.bar(results[x], results[y], color=colors[:len(results)])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.01 * yval), round(yval, 4), ha='center', va='bottom')
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_multiple_lines(results: pd.DataFrame, x: str, y_cols: list, labels: list, output_path: str, title: str = "") -> None:
    """Rysuje kilka linii na jednym wykresie (np. do zestawienia RMSE z MAE)."""
    plt.figure(figsize=(10, 6))
    for y, label in zip(y_cols, labels):
        plt.plot(results[x], results[y], marker='o', label=label)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("Wartość (Value)")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

