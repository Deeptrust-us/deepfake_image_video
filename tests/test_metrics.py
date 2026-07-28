import os
import sys
import unittest
import numpy as np

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.metrics import compute_eer, compute_frame_metrics


class TestMetrics(unittest.TestCase):
    def test_eer_perfect_prediction(self):
        # Perfect separation
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.85, 0.95, 0.88])
        eer = compute_eer(y_true, y_proba)
        self.assertAlmostEqual(eer, 0.0, places=5)

    def test_eer_random_prediction(self):
        # Random separation
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        eer = compute_eer(y_true, y_proba)
        self.assertAlmostEqual(eer, 0.5, places=5)

    def test_eer_opposite_prediction(self):
        # Opposite (completely wrong) separation
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.9, 0.85, 0.95, 0.88, 0.1, 0.2, 0.15, 0.05])
        eer = compute_eer(y_true, y_proba)
        self.assertAlmostEqual(eer, 1.0, places=5)

    def test_eer_partially_correct(self):
        # Partially correct: one error in each class
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Reals: 0.1, 0.2, 0.3, 0.7 (one high probability real)
        # Fakes: 0.8, 0.9, 0.6, 0.2 (one low probability fake)
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.6, 0.2])
        eer = compute_eer(y_true, y_proba)
        # Expected EER should be around 0.25 (1 error out of 4)
        self.assertAlmostEqual(eer, 0.25, places=2)


if __name__ == '__main__':
    unittest.main()
