import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from preprocessing import Preprocessor
from knn_imputer import KNNImputer

def test_preprocessing():
    print("Running test_preprocessing...")
    X = np.array([
        [1.0, 2.0, np.nan],
        [4.0, 5.0, 6.0],
        [np.nan, 8.0, 9.0]
    ])
    prep = Preprocessor()
    
    mask = prep.build_missing_mask(X)
    assert mask[0, 2] == True
    assert mask[2, 0] == True
    assert mask[1, 1] == False
    
    X_no_nan = np.array([
        [0.0, 10.0],
        [5.0, 20.0]
    ])
    prep.fit_minmax(X_no_nan)
    X_norm = prep.transform_minmax(X_no_nan)
    assert X_norm[0, 0] == 0.0
    assert X_norm[1, 0] == 1.0
    assert X_norm[0, 1] == 0.0
    assert X_norm[1, 1] == 1.0
    print("test_preprocessing passed.")

def test_distance():
    print("Running test_distance...")
    imputer = KNNImputer()
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 5.0, 3.0])
    c = np.array([4.0, 2.0, 7.0])
    
    dist_ab = imputer._distance(a, b, target_col=1)
    assert dist_ab == 0.0
    
    dist_ac = imputer._distance(a, c, target_col=1)
    assert np.isclose(dist_ac, np.sqrt(12.5))
    print("test_distance passed.")

def test_aggregate():
    print("Running test_aggregate...")
    imputer_uniform = KNNImputer(weights='uniform')
    neighbors = [(0.1, 10.0), (0.5, 20.0)]
    assert imputer_uniform._aggregate(neighbors) == 15.0
    
    imputer_distance = KNNImputer(weights='distance', epsilon=0)
    val = imputer_distance._aggregate(neighbors)
    assert np.isclose(val, 140.0/12.0)
    print("test_aggregate passed.")

def run_all():
    test_preprocessing()
    test_distance()
    test_aggregate()
    print("Wszystkie testy zakończyły się sukcesem!")

if __name__ == "__main__":
    run_all()
