# kNN Imputer: Imputacja brakujących wartości

Projekt realizuje algorytm uzupełniania braków w danych numerycznych (imputację) metodą **k-Nearest Neighbors (kNN)**. Algorytm na podstawie znanych wartości znajduje `k` najbardziej podobnych rekordów (sąsiadów) i wylicza brakującą wartość z wykorzystaniem średniej (lub średniej ważonej odległością).

## Struktura Katalogów

Projekt posiada budowę modularną, z jasnym podziałem odpowiedzialności:

* **`data/`** – tutaj automatycznie pobierane są pliki ze zbiorami danych (np. Adult, RNA-Seq) z serwisu OpenML. Ten folder jest ignorowany przez Gita.
* **`results/`** – miejsce, do którego skrypt `run_experiments.py` generuje tabele z wynikami `.csv` oraz narysowane wykresy `.png`.
* **`scripts/`** – skrypty uruchomieniowe. 
  * `run_experiments.py` to główny program, który przeprowadza badanie wszystkich 3 zbiorów, generując pliki raportów.
* **`src/`** – główny kod źródłowy biblioteki:
  * `knn_imputer.py` – zawiera klasę `KNNImputer`, która implementuje m.in. zignorowanie pustych komórek w trakcie liczenia odległości euklidesowej oraz agregację wyników.
  * `preprocessing.py` – obsługuje wykrywanie dziur, sztuczne ukrywanie danych (do testów) i **Normalizację Min-Max**, która jest krytyczna dla matematyki algorytmu kNN.
  * `data_loader.py` – narzędzia przygotowujące czyste Numpy Arrays (macierze z liczbami) z plików/zbiorów danych.
  * `metrics.py` – zbiór funkcji (m.in. `rmse`, `mae` oraz `measure_time`) służących do pomiarów jakości i wydajności.
  * `experiments.py` – środowisko testowe badające skalowalność czasu względem liczby cech i wierszy.
  * `visualization.py` – wrapper na pakiet `matplotlib` automatyzujący rysowanie wykresów z wyników.
* **`tests/`** – pliki weryfikacyjne dla programistów:
  * `test_logic.py` – zestaw unit testów, wywoływany przez pakiet `pytest`, sprawdzający wewnętrzną matematykę w programie.

## Jak to uruchomić?

### 1. Przygotowanie środowiska (Instalacja)
Aby uruchomić aplikację, potrzebujesz zainstalowanych pakietów wymienionych w pliku `requirements.txt`. Przejdź w terminalu do głównego folderu z projektem (tam gdzie ten README) i wpisz:

```bash
# Opcjonalnie: utwórz i aktywuj wirtualne środowisko
# python -m venv venv
# .\venv\Scripts\activate

# Zainstaluj zależności:
pip install -r requirements.txt
```

### 2. Uruchomienie głównych eksperymentów
Główny punkt programu to przebadanie skalowalności i jakości metody na zbiorach Breast Cancer, Adult oraz RNA-Seq.

Wpisz w konsoli:
```bash
python scripts/run_experiments.py
```
*Gdy skrypt skończy działać (może to potrwać długo dla największego zbioru Adult), wszystkie wyniki i wykresy pojawią się w folderze `results/`.*

### 3. Uruchamianie testów automatycznych
Jeśli chcesz się upewnić, że wewnętrzna matematyka algorytmu działa poprawnie, możesz w ułamku sekundy odpalić testy jednostkowe:

```bash
python -m pytest tests/test_logic.py
```

## Jak działa ten algorytm od środka?
Gdy każesz programowi "naprawić" uszkodzone dane z dziurami (wartościami `NaN`), algorytm pod maską robi dla każdej pustej komórki następujące rzeczy:
1. **Filtruje** potencjalnych kandydatów: Sąsiadem nie może zostać ktoś, kto sam w tej samej kolumnie ma dziurę z brakiem danych.
2. **Porównuje** cały badany wiersz ze wszystkimi prawidłowymi kandydatami: Obliczana jest tzw. modyfikowana Odległość Euklidesowa. Algorytm bierze po kolei wszystkie kolumny dla obu badanych wierszy i mierzy między nimi różnicę pod warunkiem, że obie strony ją znają (nie brakuje jej na żadnym z zestawianych rekordów).
3. **Sortuje i ucina**: Wyniki pomiarów są sortowane na zasadzie "kto ma najmniejszą różnicę, ten ląduje na początku tabeli". Następnie algorytm ucina tabelę do zaledwie pierwszych, np. $K=5$ (pięciu) pozycji. To są nasi tzw. "najbliżsi sąsiedzi".
4. **Oblicza wartość docelową (agregacja)**: Patrzy, jakie wartości miała ta znaleziona piątka sąsiadów w kolumnie, której nam brakowało, a następnie – w zależności od konfiguracji wariantu algorytmu – oddaje Ci ich zwykłą uśrednioną wartość (`uniform`) lub pozwala decydować bardziej temu "najbliższemu" (`distance`).

## 5. Interpretacja Wyników (RMSE, MAE i Wykresy)

Prawdopodobnie najczęściej zadawanym pytaniem do tego typu systemów jest: *Dlaczego nie użyliście metryk takich jak Precision i Accuracy?*
To dlatego, że nie klasyfikujemy zdjęć na psy i koty, tylko próbujemy odtworzyć wartości na skali ciągłej (problem regresji/imputacji). Kiedy nasz algorytm przewidzi `5.2` w miejscu, w którym oryginalnie brakowało `5.0`, to nie jest to błąd całkowity. Do mierzenia takich odchyleń służą:

* **RMSE (Root Mean Squared Error)**: Pierwiastek błędu średniokwadratowego. Wyjątkowo mocno "karze" algorytm za duże odstępstwa od prawdy.
* **MAE (Mean Absolute Error)**: Średni błąd bezwzględny.

### Co oznacza błąd rzędu np. 0.08?
Przed badaniami używamy w kodzie `MinMaxScaler`. Ściska on wszystkie wartości cech do przedziału `[0.0, 1.0]`. Z tego względu, błąd wielkości `0.08` dla znormalizowanych danych można interpretować bezpośrednio jako **średnią pomyłkę o ~8% skali danej cechy**. 

*  Proste wypychanie dziur średnią daje często RMSE w okolicach `0.15` (15%). Nasz kNN rzędu `0.07` (7%) to dwukrotna poprawa jakości!

### Co znajdziemy na wygenerowanych wykresach?
W ramach projektu generujemy bogaty raport graficzny, zawierający najczęstsze w profesjonalnych artykułach warianty analityczne:
1. `baseline_bar_{zbiór}.png`: Słupkowe wizualne zestawienie wyższości metody kNN nad prostackimi uzupełnianiem Średnią lub Medianą (oraz porównanie z kNN ze scikit-learn).
2. `quality_rmse_mae_{zbiór}.png`: Zestawienie dwóch najczęstszych błędów (RMSE vs MAE) i tego, jak reagują na wstrzykiwanie coraz większej puli braków danych.
3. `weights_comparison_{zbiór}.png`: Badanie mające pokazać różnicę między wagą typu `distance` a zwykłą średnią ze znalezionych sąsiadów (`uniform`). W oryginalnych pracach analitycznych zazwyczaj dowodzi się, że metoda `distance` zyskuje dużą przewagę na skomplikowanych zbiorach.
4. `scalability_k_plot_{zbiór}.png`: Wykres tego, jak zwiększanie wielkości próbki sąsiadów $K$ wydłuża fizyczny czas poszukiwań.
5. `scalability_{rows/cols}_{zbiór}.png`: Badanie krytycznego narzutu (Time Complexity) metody kNN, która wykazuje zjawisko złożoności kwadratowej. Czas wykonania rośnie lawinowo wraz ze zwiększaniem wymiarów tablicy.
